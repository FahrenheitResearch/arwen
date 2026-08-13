//! Bit-parity gate for optimization work.
//!
//! Recomputes the integer folds for every dumped real sweep and compares them
//! against a stored baseline. Any optimization that changes a single fold on
//! any gate fails here.
//!
//!   cargo run --release --example parity -- --write   # capture a baseline
//!   cargo run --release --example parity              # check against it
//!
//! Input dumps live in `$RGD_DUMPS` (default `/tmp/rgd-dumps`). Baselines live
//! in `$RGD_BASELINE` (default `/tmp/rgd-baseline`).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn dump_dirs() -> Vec<PathBuf> {
    let root = std::env::var("RGD_DUMPS").unwrap_or_else(|_| "/tmp/rgd-dumps".into());
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .expect("dump root")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            p.is_dir() && (name.starts_with("wb_") || name == "dump_keax")
        })
        .collect();
    dirs.sort();
    dirs
}

fn cuts_of(dir: &Path) -> Vec<String> {
    let mut cuts: Vec<String> = fs::read_dir(dir)
        .expect("dump dir")
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            let rest = name.strip_prefix("raw_cut")?.strip_suffix(".json")?;
            Some(rest.to_string())
        })
        .collect();
    cuts.sort();
    cuts
}

fn main() {
    let write = std::env::args().any(|a| a == "--write");
    let baseline =
        PathBuf::from(std::env::var("RGD_BASELINE").unwrap_or_else(|_| "/tmp/rgd-baseline".into()));
    if write {
        fs::create_dir_all(&baseline).expect("create baseline dir");
    }

    let mut sweeps = 0usize;
    let mut gates_total = 0usize;
    let mut mismatched = 0usize;
    let mut elapsed_ms = 0.0f64;

    for dir in dump_dirs() {
        let volume = dir.file_name().unwrap().to_str().unwrap().to_string();
        for cut in cuts_of(&dir) {
            let meta = Json::read(&dir.join(format!("raw_cut{cut}.json")));
            let rows = meta.usize("rows");
            let gates = meta.usize("gates");
            let bytes = fs::read(dir.join(format!("raw_cut{cut}.bin"))).unwrap();
            let observed: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let nyq = meta.f32_array("nyquist_mps");
            let az: Vec<f32> = meta
                .f32_array("azimuth_deg")
                .iter()
                .map(|a| a.rem_euclid(360.0))
                .collect();

            let resolved = region_global_dealias::solver::resolve_nyquist(&nyq, rows);
            let wraps = region_global_dealias::solver::sweep_wraps(&az);

            let started = Instant::now();
            let folds = region_global_dealias::solver::region_folds(
                &observed, &resolved, rows, gates, wraps,
            );
            elapsed_ms += started.elapsed().as_secs_f64() * 1000.0;

            // Folds are tiny integers; i8 keeps the baseline compact.
            let packed: Vec<i8> = folds.iter().map(|&f| f.clamp(-127, 127) as i8).collect();
            let path = baseline.join(format!("{volume}_{cut}.i8"));
            if write {
                fs::write(&path, packed.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
            } else {
                let want = fs::read(&path).unwrap_or_else(|_| {
                    panic!(
                        "missing baseline {} - run with --write first",
                        path.display()
                    )
                });
                let diff = want
                    .iter()
                    .zip(packed.iter())
                    .filter(|(a, b)| **a as i8 != **b)
                    .count();
                mismatched += diff;
                if diff > 0 {
                    println!("  MISMATCH {volume} cut{cut}: {diff} folds differ");
                }
            }
            sweeps += 1;
            gates_total += rows * gates;
        }
    }

    println!(
        "{} sweeps, {:.1} M gates, solver time {:.1} ms",
        sweeps,
        gates_total as f64 / 1e6,
        elapsed_ms
    );
    if write {
        println!("baseline written to {}", baseline.display());
    } else if mismatched == 0 {
        println!("PARITY OK - 0 folds differ");
    } else {
        println!("PARITY FAILED - {mismatched} folds differ");
        std::process::exit(1);
    }
}

/// Minimal JSON field reader so the example needs no dependency.
struct Json(String);

impl Json {
    fn read(path: &Path) -> Self {
        Json(fs::read_to_string(path).unwrap())
    }
    fn usize(&self, key: &str) -> usize {
        let pat = format!("\"{key}\":");
        let i = self.0.find(&pat).unwrap() + pat.len();
        self.0[i..]
            .trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap()
    }
    fn f32_array(&self, key: &str) -> Vec<f32> {
        let pat = format!("\"{key}\":[");
        let i = self.0.find(&pat).unwrap() + pat.len();
        let j = self.0[i..].find(']').unwrap() + i;
        self.0[i..j]
            .split(',')
            .map(|v| v.trim().parse::<f32>().unwrap_or(f32::NAN))
            .collect()
    }
}
