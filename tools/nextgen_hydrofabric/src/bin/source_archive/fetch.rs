//! HTTPS transport only is delegated to system curl. Rust bounds, hashes and
//! publishes every byte; no shell, decoder, Python or unbounded download buffer.
use super::check_archive;
use nextgen_hydrofabric::Result;
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        // This directory was created exclusively by this job; never an old
        // cache, another job's scratch, or a user's archive.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn transfer(url: &str, target: &Path, bytes: u64, expected: &str) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)?;
    let mut child = Command::new("curl")
        .args([
            "--disable", // Ignore ~/.curlrc, including unexpected output paths.
            "--silent",
            "--show-error",
            "--fail",
            "--location",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--connect-timeout",
            "20",
            "--max-time",
            "180",
            "--max-redirs",
            "5",
            "--url",
            url,
        ])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("HTTPS acquisition requires system curl: {e}"))?;
    // Always reap even on disk/read/size failures. At most pinned bytes are
    // written to disk; a peer's Content-Length never determines the bound.
    let streamed = (|| -> Result<(u64, String)> {
        let mut stream = child.stdout.take().ok_or("missing curl stdout")?;
        let mut buffer = [0u8; 65536];
        let mut count = 0u64;
        let mut hash = Sha256::new();
        loop {
            let n = stream.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            count = count
                .checked_add(n as u64)
                .ok_or("download byte count overflow")?;
            if count > bytes {
                return Err("download exceeds pinned compressed size".into());
            }
            file.write_all(&buffer[..n])?;
            hash.update(&buffer[..n]);
        }
        file.sync_all()?;
        Ok((count, format!("{:x}", hash.finalize())))
    })();
    if streamed.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    let (count, actual) = streamed?;
    if !status.success() {
        return Err(format!("HTTPS transfer failed: {status}").into());
    }
    if count != bytes || actual != expected {
        return Err("download SHA256/length mismatch".into());
    }
    Ok(())
}

fn one(item: &Value) -> Result<Value> {
    let started = Instant::now();
    let path = Path::new(item["path"].as_str().ok_or("missing archive path")?);
    let bytes = item["bytes"].as_u64().ok_or("missing archive bytes")?;
    let expected = item["sha256"].as_str().ok_or("missing archive SHA256")?;
    let url = item["url"].as_str().ok_or("missing archive URL")?;
    if check_archive(path, bytes, expected)? {
        return Ok(json!({"path":path,"status":"concurrent_reuse","downloaded_bytes":0}));
    }
    let root = path.parent().ok_or("missing archive root")?;
    std::fs::create_dir_all(root)?;
    let name = path
        .file_name()
        .ok_or("missing archive name")?
        .to_string_lossy();
    let tmp = root.join(format!(
        ".{name}.fetch-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    std::fs::create_dir(&tmp)?;
    let scratch = Scratch(tmp);
    let partial = scratch.0.join("archive.partial");
    // Retry transport failures from a fresh bounded file, never concatenate a
    // retry onto partial stdout. Hash/size failures are returned immediately.
    let mut attempts = 0;
    loop {
        attempts += 1;
        match transfer(url, &partial, bytes, expected) {
            Ok(()) => break,
            Err(e) if attempts < 3 && e.to_string().starts_with("HTTPS transfer failed:") => {
                std::fs::remove_file(&partial)?;
            }
            Err(e) => return Err(format!("archive {}: {e}", path.display()).into()),
        }
    }
    // Same-filesystem hard link is atomic and fails if a destination exists.
    // Unlike rename it can never overwrite a concurrently published archive.
    let publication = match std::fs::hard_link(&partial, path) {
        Ok(()) => "downloaded",
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            if !check_archive(path, bytes, expected)? {
                return Err("concurrent archive disappeared before verification".into());
            }
            "concurrent_reuse"
        }
        Err(e) => return Err(e.into()),
    };
    Ok(
        json!({"path":path,"status":publication,"downloaded_bytes":bytes,
        "sha256":expected,"attempts":attempts,
        "wall_ms":started.elapsed().as_secs_f64()*1000.0}),
    )
}

pub(super) fn missing(plan: &Value) -> Result<Value> {
    let started = Instant::now();
    let missing = plan["cache"]["required_downloads"]
        .as_array()
        .ok_or("missing verified cache plan")?;
    // Complete preflight runs before any side effect. Other independent jobs
    // may finish when one fails; their verified cache files remain reusable.
    let results = missing.par_iter().map(one).collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"archives":results,"rayon_workers":rayon::current_num_threads(),
        "parallelism":"one missing archive per Rayon job; HTTPS curl transport; Rust bounded streaming SHA256, 65536-byte buffer per job",
        "wall_ms":started.elapsed().as_secs_f64()*1000.0}),
    )
}
