//! Bind the Zarr reader and its in-tree numerical dependencies to the release.
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=GPUWM_BRIDGE_SOURCE_REV");
    for path in ["src", "build.rs", "Cargo.toml", "Cargo.lock", ".cargo", "vendor",
                 "../rustwx/crates/netcdf-writer", "../rw_wps"] {
        println!("cargo:rerun-if-changed={path}");
    }
    let revision = std::env::var("GPUWM_BRIDGE_SOURCE_REV")
        .ok().filter(|value| is_commit(value)).or_else(clean_revision)
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=GPUWM_BRIDGE_SOURCE_REV={revision}");
}

fn is_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn git(args: &[&str]) -> Option<String> {
    let directory = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let result = Command::new("git").arg("-C").arg(directory).args(args).output().ok()?;
    result.status.success().then_some(())?;
    Some(String::from_utf8(result.stdout).ok()?.trim().to_owned())
}

fn clean_revision() -> Option<String> {
    for name in ["HEAD", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", &reference]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    // Path dependencies enter this executable too. Untracked source files
    // must not allow a new crate to impersonate the old committed tree.
    if !git(&["status", "--porcelain", "--untracked-files=normal", "--", ".",
              "../rustwx/crates/netcdf-writer", "../rw_wps"])?.is_empty() {
        return None;
    }
    let revision = git(&["rev-parse", "HEAD"])?;
    is_commit(&revision).then_some(revision)
}
