// Records the source revision in the binary (fabric_metadata.builder_revision).
use std::process::Command;
fn main() {
    let rev = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|r| r.status.success())
        .map(|r| String::from_utf8_lossy(&r.stdout).trim().to_owned())
        .unwrap_or_else(|| "unrecorded-source".into());
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--", "."])
        .output()
        .ok()
        .filter(|r| r.status.success())
        .map(|r| !r.stdout.is_empty())
        .unwrap_or(false);
    println!("cargo:rustc-env=HYDRO_WORLD_REV={rev}{}", if dirty { "+dirty" } else { "" });
    // The release pin step reads GPUWM_BRIDGE_SOURCE_REV=<commit> from every bundled binary.
    println!("cargo:rerun-if-env-changed=GPUWM_BRIDGE_SOURCE_REV");
    let stamp = std::env::var("GPUWM_BRIDGE_SOURCE_REV")
        .ok()
        .or_else(|| (!dirty).then(|| rev.clone()))
        .filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=GPUWM_BRIDGE_SOURCE_REV={stamp}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    // HEAD of a worktree only names the branch; the branch ref (or packed-refs)
    // is what changes on a commit, so watch those too.
    let path = |p: &str| Command::new("git").args(["rev-parse", "--git-path", p]).output().ok().map(|r| String::from_utf8_lossy(&r.stdout).trim().to_owned());
    for p in [Some("HEAD".to_string()), Some("packed-refs".to_string())].into_iter().flatten() {
        if let Some(x) = path(&p) {
            println!("cargo:rerun-if-changed={x}");
        }
    }
    if let Ok(r) = Command::new("git").args(["symbolic-ref", "-q", "HEAD"]).output() {
        if r.status.success() {
            if let Some(x) = path(String::from_utf8_lossy(&r.stdout).trim()) {
                println!("cargo:rerun-if-changed={x}");
            }
        }
    }
}
