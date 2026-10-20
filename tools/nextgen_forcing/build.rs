// SPDX-License-Identifier: Apache-2.0
// The release bridge verifier reads this literal without executing the library.
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=GPUWM_BRIDGE_SOURCE_REV");
    for path in [
        "src",
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "../rustwx/vendor/netcrust",
        "../rustwx/crates/netcdf-writer",
        "../rustwx/Cargo.toml",
        "../rustwx/Cargo.lock",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    let git = |args: &[&str]| -> Option<String> {
        let output = Command::new("git").args(args).output().ok()?;
        output.status.success().then_some(())?;
        Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
    };
    // Include dependency edits outside this crate in the clean-tree check.
    let root = git(&["rev-parse", "--show-toplevel"]);
    for name in ["HEAD", "packed-refs"] {
        if let Some(path) = git(&["rev-parse", "--path-format=absolute", "--git-path", name]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git(&[
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            &reference,
        ]) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    let clean = root
        .as_deref()
        .and_then(|root| git(&["-C", root, "status", "--porcelain", "--untracked-files=no"]))
        .is_some_and(|status| status.is_empty());
    let revision = std::env::var("GPUWM_BRIDGE_SOURCE_REV")
        .ok()
        .or_else(|| clean.then(|| git(&["rev-parse", "HEAD"])).flatten())
        .filter(|value| {
            value.len() == 40
                && value
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=GPUWM_BRIDGE_SOURCE_REV={revision}");
}
