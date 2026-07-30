//! Compile-time build stamp: RW_BUILD_SHA carries the short git SHA (plus
//! `-dirty` when the tree has local changes) into every store written by
//! rw_ingest, so run.json records exactly which build produced it.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    // HEAD alone goes stale: same-branch commits rewrite the ref file (not
    // HEAD), so also watch the ref HEAD points at, plus the index so
    // staged/dirty transitions refresh the -dirty flag.
    if let Ok(head) = std::fs::read_to_string("../../.git/HEAD") {
        if let Some(reference) = head.strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed=../../.git/{}", reference.trim());
        }
    }
    println!("cargo:rerun-if-changed=../../.git/index");
    let sha = build_sha().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=RW_BUILD_SHA={sha}");
}

fn build_sha() -> Option<String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let rev = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .current_dir(&manifest_dir)
        .output()
        .ok()?;
    if !rev.status.success() {
        return None;
    }
    let mut sha = String::from_utf8(rev.stdout).ok()?.trim().to_string();
    if sha.is_empty() {
        return None;
    }
    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(&manifest_dir)
        .output()
        .ok()?;
    if !status.status.success() {
        return None;
    }
    if status.stdout.iter().any(|byte| !byte.is_ascii_whitespace()) {
        sha.push_str("-dirty");
    }
    Some(sha)
}
