use std::{path::Path, process::Command};
fn valid(v: &str) -> bool {
    v.len() == 40
        && v.bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}
fn main() {
    println!("cargo:rerun-if-env-changed=GPUWM_BRIDGE_SOURCE_REV");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=data");
    let env = std::env::var("GPUWM_BRIDGE_SOURCE_REV")
        .ok()
        .filter(|v| valid(v));
    let root = std::env::var("CARGO_MANIFEST_DIR")
        .ok()
        .and_then(|v| Path::new(&v).parent()?.parent().map(|p| p.to_path_buf()));
    let revision = env
        .or_else(|| {
            let root = root?;
            for item in ["HEAD", "packed-refs"] {
                if let Ok(path) = Command::new("git")
                    .arg("-C")
                    .arg(&root)
                    .args(["rev-parse", "--path-format=absolute", "--git-path", item])
                    .output()
                {
                    if path.status.success() {
                        println!(
                            "cargo:rerun-if-changed={}",
                            String::from_utf8_lossy(&path.stdout).trim()
                        );
                    }
                }
            }
            if let Ok(reference) = Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(["symbolic-ref", "-q", "HEAD"])
                .output()
            {
                if reference.status.success() {
                    if let Ok(path) = Command::new("git")
                        .arg("-C")
                        .arg(&root)
                        .args([
                            "rev-parse",
                            "--path-format=absolute",
                            "--git-path",
                            String::from_utf8_lossy(&reference.stdout).trim(),
                        ])
                        .output()
                    {
                        if path.status.success() {
                            println!(
                                "cargo:rerun-if-changed={}",
                                String::from_utf8_lossy(&path.stdout).trim()
                            );
                        }
                    }
                }
            }
            let dirty = Command::new("git")
                .arg("-C")
                .arg(&root)
                .args([
                    "status",
                    "--porcelain",
                    "-uno",
                    "--",
                    ".",
                    "../grib1_bridge",
                ])
                .output()
                .ok()?;
            if !dirty.status.success() || !dirty.stdout.is_empty() {
                return None;
            }
            let head = Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()?;
            if !head.status.success() {
                return None;
            }
            let value = String::from_utf8(head.stdout).ok()?.trim().to_string();
            valid(&value).then_some(value)
        })
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=GPUWM_BRIDGE_SOURCE_REV={revision}");
}
