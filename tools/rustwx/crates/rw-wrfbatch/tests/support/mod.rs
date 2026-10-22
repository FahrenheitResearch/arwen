//! Shared integration-test scratch fixture.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(prefix: &str, tag: &str) -> Self {
        // Parallel tests shared a clock tick on the 2.8.6 Windows runner.
        // Keep the counter so one test's Drop cannot remove another's root.
        static NEXT_SCRATCH: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or_default();
        let dir = std::env::temp_dir().join(format!(
            "{prefix}-{tag}-{}-{nonce}-{}",
            std::process::id(),
            NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Self(dir)
    }

    #[allow(dead_code)]
    pub fn path(&self) -> &Path { &self.0 }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
