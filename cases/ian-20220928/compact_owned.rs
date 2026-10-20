//! Consolidate byte-identical head/export twins in an agent-owned prep tree.
//! No artifacts are removed: both sealed paths keep identical bytes.
use std::{
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::Instant,
};
fn pairs(head: &Path, export: &Path, result: &mut Vec<(PathBuf, PathBuf)>) {
    for entry in fs::read_dir(head).unwrap() {
        let entry = entry.unwrap();
        let a = entry.path();
        let b = export.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            if b.is_dir() {
                pairs(&a, &b, result);
            }
        } else if b.is_file() && entry.metadata().unwrap().len() == b.metadata().unwrap().len() {
            result.push((a, b));
        }
    }
}
fn identical(a: &Path, b: &Path) -> bool {
    let (mut a, mut b) = (
        BufReader::new(fs::File::open(a).unwrap()),
        BufReader::new(fs::File::open(b).unwrap()),
    );
    let (mut x, mut y) = ([0_u8; 65536], [0_u8; 65536]);
    loop {
        let n = a.read(&mut x).unwrap();
        let m = b.read(&mut y).unwrap();
        if n != m || x[..n] != y[..m] {
            return false;
        }
        if n == 0 {
            return true;
        }
    }
}
fn main() {
    use std::os::unix::fs::MetadataExt;
    let start = Instant::now();
    let root = PathBuf::from(std::env::args().nth(1).expect("prepared root"));
    let mut work = vec![];
    pairs(
        &root.join("hierarchy-head"),
        &root.join("hierarchy-artifacts"),
        &mut work,
    );
    let next = AtomicUsize::new(0);
    let saved = AtomicU64::new(0);
    let files = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().unwrap().get();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some((a, b)) = work.get(k) else { break };
                    if a.metadata().unwrap().ino() == b.metadata().unwrap().ino()
                        || !identical(a, b)
                    {
                        continue;
                    }
                    let tmp = b.with_extension("compact-pending");
                    fs::hard_link(a, &tmp).unwrap();
                    fs::rename(tmp, b).unwrap();
                    saved.fetch_add(a.metadata().unwrap().len(), Ordering::Relaxed);
                    files.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    println!(
        "{{\"stage\":\"byte-verified immutable hardlink consolidation\",\"threads\":{threads},\"files\":{},\"saved_bytes\":{},\"wall_seconds\":{:.3}}}",
        files.load(Ordering::Relaxed),
        saved.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
}
