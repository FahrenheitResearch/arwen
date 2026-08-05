//! The library face of ArWen's NEXRAD front door.
//!
//! `rw_nexrad` began as a bin-only crate, and two of its three modules turned
//! out to be about *archives* rather than about radar:
//!
//! * [`s3`] is an anonymous-S3 client — `ListObjectsV2` over a hardened,
//!   schema-checked XML reader, a content-addressed download cache, SHA-256
//!   receipts, and RFC-3339 time parsing. Nothing in it knows what a sweep is.
//! * [`pack`] carries the `.rwg`-derived container discipline — an 8-byte
//!   magic, a `u32` version, a length-prefixed JSON metadata block, and one
//!   contiguous little-endian payload a Python reader takes with
//!   `numpy.frombuffer`.
//!
//! Every later observation front door needs the first and imitates the
//! second, and the alternative to exposing them is a second S3 client. The
//! XML reader in particular was hardened over a nine-round fix-then-attack
//! audit; a parallel implementation would re-earn those bugs rather than
//! inherit the fixes. So the modules become public and the binary consumes
//! them exactly as it did when they were private — `main.rs` swapped three
//! `mod` lines for one `use`, and no item moved, changed signature, or
//! changed behavior.
//!
//! [`decode`] and [`live`] are genuinely NEXRAD-specific and are exported only
//! so the binary can keep reaching them through the same path as their
//! siblings. [`live`] in particular addresses [`s3`] and [`pack`] as
//! `crate::s3` and `crate::pack`, so it belongs in the crate that defines
//! them rather than in the binary that consumes them.

pub mod decode;
pub mod live;
pub mod pack;
pub mod s3;
