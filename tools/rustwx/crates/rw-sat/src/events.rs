//! Typed progress events and the error type for the follow engine —
//! mirroring `rw_ingest::events` (`IngestEvent` / `IngestError`): library
//! code emits events through a sink closure, bins print them, UI hosts
//! forward them over a channel and repaint.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use chrono::{DateTime, Utc};

/// One progress event from the GOES follow / ingest flow.
#[derive(Debug, Clone)]
pub enum SatEvent {
    /// One poll cycle started for a band watcher.
    PollStarted { band: u8, prefixes: Vec<String> },
    /// One poll cycle finished; `new_keys` is the number of previously
    /// unseen objects discovered across the polled prefixes.
    PollDone { band: u8, new_keys: usize, ms: u128 },
    /// An object download began (`bytes` is the listed S3 size).
    DownloadStarted { key: String, bytes: u64 },
    DownloadDone {
        key: String,
        bytes: u64,
        ms: u128,
        cache_hit: bool,
    },
    /// One frame landed in the store.
    FrameWritten {
        model: String,
        run: String,
        hhmm: u16,
        scan_time_utc: DateTime<Utc>,
        path: PathBuf,
        bytes: u64,
        encode_ms: u64,
    },
    /// The rolling window evicted old frames.
    Evicted {
        model: String,
        frames: usize,
        bytes: u64,
    },
    /// The next poll was delayed (jitter/backoff included), for UIs that
    /// show a countdown.
    Sleeping { ms: u64 },
    Info { message: String },
    Warning { message: String },
}

/// The bins' sink: human-readable lines, `Info`/frame/download events to
/// stdout, warnings to stderr.
pub fn print_event(event: &SatEvent) {
    match event {
        SatEvent::PollStarted { band, prefixes } => {
            println!("poll C{band:02}: {}", prefixes.join(" + "));
        }
        SatEvent::PollDone { band, new_keys, ms } => {
            println!("poll C{band:02}: {new_keys} new object(s) in {ms} ms");
        }
        SatEvent::DownloadStarted { key, bytes } => {
            println!("get {key} ({bytes} bytes)");
        }
        SatEvent::DownloadDone {
            key,
            bytes,
            ms,
            cache_hit,
        } => {
            let cached = if *cache_hit { " (cache hit)" } else { "" };
            println!("got {key} ({bytes} bytes, {ms} ms){cached}");
        }
        SatEvent::FrameWritten {
            model,
            run,
            hhmm,
            scan_time_utc,
            path,
            bytes,
            encode_ms,
        } => {
            println!(
                "frame {model}/{run}/t{hhmm:04} scan {} -> {} ({bytes} bytes, encode {encode_ms} ms)",
                scan_time_utc.format("%Y-%m-%dT%H:%M:%SZ"),
                path.display()
            );
        }
        SatEvent::Evicted {
            model,
            frames,
            bytes,
        } => {
            println!("evicted {frames} frame(s) / {bytes} bytes from {model}");
        }
        SatEvent::Sleeping { ms } => {
            println!("sleeping {} s", *ms as f64 / 1000.0);
        }
        SatEvent::Info { message } => println!("{message}"),
        SatEvent::Warning { message } => eprintln!("{message}"),
    }
}

/// A cancel flag that is never set, for callers without cancellation.
pub static NEVER_CANCEL: AtomicBool = AtomicBool::new(false);

/// Errors from the follow flow. `Cancelled` is the variant callers match on
/// (the cancel flag was observed at a boundary); everything else passes
/// through with its original message.
#[derive(Debug, thiserror::Error)]
pub enum SatError {
    #[error("goes follow cancelled")]
    Cancelled,
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl SatError {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, SatError::Cancelled)
    }
}

/// Internal shorthand: wrap any error into [`SatError::Other`].
pub(crate) fn other(err: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> SatError {
    SatError::Other(err.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_is_distinguishable_and_displays() {
        let cancelled = SatError::Cancelled;
        assert!(cancelled.is_cancelled());
        assert_eq!(cancelled.to_string(), "goes follow cancelled");

        let wrapped = other("bucket unreachable");
        assert!(!wrapped.is_cancelled());
        assert_eq!(wrapped.to_string(), "bucket unreachable");
    }
}
