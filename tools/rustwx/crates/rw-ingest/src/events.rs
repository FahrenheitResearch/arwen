//! Progress events, stages, and the ingest error type.
//!
//! The per-hour flow used to print progress straight to stdout/stderr; as a
//! library it emits [`IngestEvent`]s through the sink on
//! [`IngestConfig`](crate::IngestConfig) instead. The bins pass
//! [`print_event`], which reproduces the historical lines byte-for-byte
//! (manifests and smoke flows are pinned to them); UI hosts forward events
//! over a channel and repaint.

use std::sync::atomic::AtomicBool;

/// One stage of the per-hour ingest flow, in execution order. Fetch stages
/// run on the network half ([`fetch_hour`](crate::fetch_hour)); the rest on
/// the CPU half ([`process_fetched_hour`](crate::process_fetched_hour)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IngestStage {
    FetchPrs,
    FetchSfc,
    ExtractPrs,
    ExtractSfc,
    ThermoDecode,
    Derived,
    Heavy,
    Write,
    Verify,
}

impl IngestStage {
    /// Every stage, in execution order.
    pub const ALL: [IngestStage; 9] = [
        IngestStage::FetchPrs,
        IngestStage::FetchSfc,
        IngestStage::ExtractPrs,
        IngestStage::ExtractSfc,
        IngestStage::ThermoDecode,
        IngestStage::Derived,
        IngestStage::Heavy,
        IngestStage::Write,
        IngestStage::Verify,
    ];

    /// Short human label for progress UIs.
    pub fn label(self) -> &'static str {
        match self {
            IngestStage::FetchPrs => "fetch prs",
            IngestStage::FetchSfc => "fetch sfc",
            IngestStage::ExtractPrs => "extract prs",
            IngestStage::ExtractSfc => "extract sfc",
            IngestStage::ThermoDecode => "thermo decode",
            IngestStage::Derived => "derived",
            IngestStage::Heavy => "heavy",
            IngestStage::Write => "write",
            IngestStage::Verify => "verify",
        }
    }
}

/// One progress event from the per-hour flow. `Info`/`Warning` messages are
/// complete lines formatted exactly as the bins have always printed them
/// (`Info` -> stdout, `Warning` -> stderr); stage events bracket each
/// stage's wall time and carry no text.
#[derive(Debug, Clone)]
pub enum IngestEvent {
    StageStarted {
        hour: u16,
        stage: IngestStage,
    },
    StageDone {
        hour: u16,
        stage: IngestStage,
        ms: u128,
    },
    /// Historical stdout line (e.g. the heavy per-kernel breakdown, the
    /// verify-ok line, profile-skip notes).
    Info {
        hour: u16,
        message: String,
    },
    /// Historical stderr line (missing-plane skips, fallbacks, degraded
    /// stages).
    Warning {
        hour: u16,
        message: String,
    },
}

/// The sink the bins use: print `Info` to stdout and `Warning` to stderr,
/// verbatim — byte-identical to the historical inline prints. Stage events
/// are dropped (the bins print their own per-hour summary lines).
pub fn print_event(event: IngestEvent) {
    match event {
        IngestEvent::Info { message, .. } => println!("{message}"),
        IngestEvent::Warning { message, .. } => eprintln!("{message}"),
        IngestEvent::StageStarted { .. } | IngestEvent::StageDone { .. } => {}
    }
}

/// A cancel flag that is never set, for callers without cancellation
/// (the bins). Do not store through this reference.
pub static NEVER_CANCEL: AtomicBool = AtomicBool::new(false);

/// Errors from the per-hour flow. `Cancelled` is the only variant callers
/// usually match on (the cancel flag was observed at a stage boundary);
/// everything else passes through with its original message so the bins'
/// error output stays identical.
#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("ingest cancelled")]
    Cancelled,
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl IngestError {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, IngestError::Cancelled)
    }
}

/// Internal shorthand: wrap any error into [`IngestError::Other`].
pub(crate) fn other(err: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> IngestError {
    IngestError::Other(err.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_is_distinguishable_and_displays() {
        let cancelled = IngestError::Cancelled;
        assert!(cancelled.is_cancelled());
        assert_eq!(cancelled.to_string(), "ingest cancelled");

        let wrapped = other("disk full");
        assert!(!wrapped.is_cancelled());
        // Transparent: the inner message passes through unchanged, so bin
        // error output stays byte-identical.
        assert_eq!(wrapped.to_string(), "disk full");
    }

    #[test]
    fn stage_order_and_labels_are_stable() {
        assert_eq!(IngestStage::ALL.len(), 9);
        assert_eq!(IngestStage::ALL[0].label(), "fetch prs");
        assert_eq!(IngestStage::ALL[8].label(), "verify");
    }
}
