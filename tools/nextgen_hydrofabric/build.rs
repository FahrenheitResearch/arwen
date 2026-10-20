use std::process::Command;
fn main() {
    // The release builds (tipbuild, the cut's host battery) build a tree that is not this repository's checkout
    // and name its commit in GPUWM_BRIDGE_SOURCE_REV, as every other gpuwm bridge reads it; git is the fallback.
    println!("cargo:rerun-if-env-changed=GPUWM_BRIDGE_SOURCE_REV");
    let rev = std::env::var("GPUWM_BRIDGE_SOURCE_REV")
        .ok()
        .filter(|v| v.len() == 40 && v.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
        .or_else(|| {
            Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|r| r.status.success())
                .map(|r| String::from_utf8_lossy(&r.stdout).trim().to_owned())
        })
        .unwrap_or_else(|| "unrecorded-source".into());
    println!("cargo:rustc-env=GPUWM_HYDROFABRIC_REV={rev}");
    println!("cargo:rerun-if-changed=build.rs");
    if let Ok(r) = Command::new("git")
        .args(["rev-parse", "--git-path", "HEAD"])
        .output()
    {
        println!(
            "cargo:rerun-if-changed={}",
            String::from_utf8_lossy(&r.stdout).trim()
        );
    }
    if let Ok(r) = Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .output()
    {
        if r.status.success() {
            if let Ok(p) = Command::new("git")
                .args([
                    "rev-parse",
                    "--git-path",
                    String::from_utf8_lossy(&r.stdout).trim(),
                ])
                .output()
            {
                println!(
                    "cargo:rerun-if-changed={}",
                    String::from_utf8_lossy(&p.stdout).trim()
                );
            }
        }
    }
}
