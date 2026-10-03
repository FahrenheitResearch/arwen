//! NEXRAD Archive II / Level II decoder entry points.
//!
//! The `read_*` functions decode into the FM301 model
//! ([`recast_radar_core::model::Volume`], `docs/design/fm301-model.md`): one
//! sweep per elevation cut in acquisition order, fields named as xradar names
//! them (REF→`DBZH`, VEL→`VRADH`, SW→`WRADH`, ZDR, PHI→`PHIDP`, RHO→`RHOHV`,
//! CFP→`CCORH`), raw `u8`/`u16` gates with the ICD coding
//! ([`recast_radar_core::model::IntCoding::nexrad`]) and each moment in its
//! native gate geometry on the sweep range. `Sweep::fixed_angle_deg` is the
//! VCP cut angle from Message 5 when the file has one (what xradar and
//! Py-ART report), else the opening radial's elevation. A moment whose gates
//! cannot share the sweep range (garbage radials of misframed files) is
//! dropped for that sweep.
//!
//! Unsupported records stay non-fatal so an app can inspect partially decoded
//! volumes while the edge-case corpus grows.
//!
//! Other radar formats live in their own crates (`recast-radar-io-odim`,
//! `-io-cfradial`, `-io-dorade`, `-io-jma`); `recast-radar-io` routes byte
//! buffers of unknown format to the right decoder, and
//! `recast-radar-io-level3` decodes Level III products (including the VAD Wind
//! Profile, `recast_radar_io_level3::vwp`).
//! [`messages`] walks the Level II message stream and decodes the message
//! bodies (message 1 radials are left to the volume decoder).
//! [`read_volume_with_metadata`] ([`metadata`]) returns the decoded volume
//! together with its metadata messages and per-sweep message 31 constant
//! blocks.
//!
//! Every value the decoders parse also reaches the volume itself: the
//! per-radial values (radial status, azimuth number, message header, RAD
//! noise levels and calibration constants, VOL transmitter powers, ...) as
//! per-ray sweep variables and FM301 monitoring variables, each sweep's
//! first-radial VOL and ELV constants as sweep attributes, and every
//! metadata message and every non-radial message header as `nexrad_*` root
//! variables ([`passthrough`]; `docs/level2/messages.md`, "Metadata in the
//! model").
//!
//! `Volume::time_reference` is the earliest radial's collection time
//! floored to the second: the first radial's in every NEXRAD file, and an
//! earlier one in converted files that store the lowest cut first although
//! it was observed last, so no ray time is negative.
//!
//! # Compression
//!
//! LDM block-bzip2 records and whole-file bzip2 volumes are decoded by
//! `recast-radar-bzip2` (this repository's decoder without unsafe code, one
//! reusable decoder per thread; the `paired-bzip2` feature decodes two
//! records at a time on one thread). Decoded record buffers are recycled
//! through a bounded process-wide pool. gzip volumes are inflated by
//! [`gzip`]: whole-buffer input in one pass into a buffer presized from the
//! gzip trailer, streaming input through a reader; both decode every member
//! of a multi-member file and ignore bytes after the last member. A gzip or
//! whole-file bzip2 copy of an LDM-compressed file is expanded whole (about
//! the size of the file inside) and its records decoded as the unwrapped
//! file's are. An LDM-compressed file cut short inside a record decodes to
//! the records before it, as a damaged record does (the first record cut
//! short is a [`NexradError::Truncated`] error).
//!
//! # Limits
//!
//! Allocations never follow header values past these caps (shared values in
//! [`recast_radar_core::bounded_read`]):
//!
//! - **Expanded input**: gzip, whole-file bzip2, and uncompressed buffers at
//!   most `MAX_DECODED_RADAR_BYTES` (512 MiB). LDM block-bzip2 volumes: at
//!   most 4,096 blocks, 16 MiB per decompressed block, 512 MiB in total.
//!   Violations are [`NexradError::Compression`] errors.
//! - **Messages**: sizes are 16-bit halfword counts (at most 131,070 bytes);
//!   every block pointer and gate array is checked against its message.
//! - **Gates**: a Message 1 row or Message 31 moment block may declare at
//!   most `MAX_GATES_PER_RADIAL` (16,384) gates; real volumes reach 1,840.
//! - **Cuts**: at most `MAX_SWEEPS_PER_VOLUME` (1,024) elevation cuts; real
//!   volumes reach 23.
//! - **Decoded moments and ray tables**: field reservations and growth are
//!   checked against a `DecodeBudget` of `MAX_DECODED_VOLUME_BYTES` (1 GiB)
//!   per volume before allocating; real volumes need at most 80 MiB. Every
//!   radial is charged 96 bytes for its coordinates and per-ray columns; a
//!   sweep whose radials differ in a value kept once per sweep (the VOL and
//!   ELV constants, spacing, indexing, identifier, moment header values) is
//!   charged for each ray's copy (at most about 150 bytes more per radial).
//! - **Metadata messages**: the volume keeps at most 1,024 fixed frames of
//!   non-radial messages (a metadata record has 134) and 1,024
//!   variable-length ones of at most 16 MiB together (charged to the
//!   `DecodeBudget`); RDA log data
//!   (message 33) inflates to at most 64 MiB per volume, all logs together,
//!   charged to the `DecodeBudget`; each of messages 3, 5 or 7, 8, 13, 15,
//!   18 and 32 keeps at most [`passthrough::MAX_COPIES_PER_MESSAGE_TYPE`]
//!   distinct copies. What goes past these caps is counted in root
//!   variables ([`passthrough`]), not reported as an error.
//!
//! Gate, cut, and budget violations are [`NexradError::LimitExceeded`]
//! errors. Preview callbacks receive a sealed copy of the partial volume, so
//! memory briefly doubles while one runs. [`read_volume_from_path`] reads the
//! whole file first; callers choose which files to open.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

mod builder;
mod bzip2_prefix;
mod fm301_attrs;
pub mod gzip;
pub mod messages;
pub mod metadata;
pub mod passthrough;
mod radial_extras;
#[cfg(feature = "write")]
pub mod write;

pub use metadata::{
    NexradMetadata, NexradVolume, RadialConstants, SweepElevationData, read_volume_with_metadata,
};
#[cfg(feature = "write")]
pub use write::{WriteError, WriteOptions, write_volume, write_volume_to};

use std::cell::RefCell;
use std::fs;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};

use chrono::{DateTime, TimeZone, Utc};
use rayon::prelude::*;
use recast_radar_core::bounded_read::{self, DecodeBudget, MAX_DECODED_RADAR_BYTES};
use recast_radar_core::model::Volume;
use thiserror::Error;

use crate::builder::{BlockGates, MomentBlock, MomentHeaderExtras, MomentPayload, VolumeBuilder};
use crate::radial_extras::{
    ElevationConstants, LegacyConstants, RadialBlockConstants, RadialExtras, RadialMessage,
    RadialMessageHeader, VolumeConstants,
};
const VOLUME_HEADER_LEN: usize = 24;
const CONTROL_WORD_LEN: usize = 12;
const MESSAGE_HEADER_LEN: usize = 16;
const RECORD_BYTES: usize = 2432;
const MSG_31_HEADER_LEN: usize = 72;
const MSG_1_HEADER_LEN: usize = 100;
const GENERIC_DATA_BLOCK_LEN: usize = 28;
const VOLUME_CONSTANT_BLOCK_LEN: usize = 44;
/// VOL block size from Build 20.0, with the ZDR bias estimate.
const VOLUME_CONSTANT_BLOCK_ZDR_BIAS_LEN: usize = 52;
const ELEVATION_CONSTANT_BLOCK_LEN: usize = 12;
const RADIAL_CONSTANT_BLOCK_LEN: usize = 20;
/// RAD block size from Build 14.0, with the calibration constants.
const RADIAL_CONSTANT_BLOCK_CALIBRATED_LEN: usize = 28;
const HALF_DEGREE_RADIALS_PER_CUT: usize = 720;
const ONE_DEGREE_RADIALS_PER_CUT: usize = 360;
const FALLBACK_RADIALS_PER_CUT: usize = 760;
const MAX_MESSAGE_31_MOMENTS: usize = 10;
/// LDM block-bzip chunks are normally a few hundred KiB decoded. Keep a
/// separate per-block ceiling so many workers cannot each inflate an
/// attacker-controlled block all the way to the full-volume budget.
const MAX_BZIP_BLOCK_DECODED_BYTES: usize = 16 * 1024 * 1024;
const MAX_BZIP_BLOCKS: usize = 4096;
/// Decoded LDM block buffers kept for reuse by later blocks and later
/// volumes. Operational blocks decode to about 0.3-1.4 MiB; a one-thread
/// decode cycles two or three buffers, so the pool removes the
/// allocate-and-page-fault cost of ~100 fresh block buffers per volume.
///
/// The bound is small on purpose. On a thread pool the decoded blocks
/// waiting for the parser can be most of the volume (the parser is the
/// slower side from about a dozen threads on), and as the parser drains
/// them every buffer it hands back would stay in a larger pool, resident
/// beside the finished volume: with 64 buffers that kept 40-50 MiB more
/// at the peak of a KTLX 2024 decode on the default pool
/// (docs/perf/cross-library.md, "Level II peak RSS against the nexrad
/// crate"). Buffers past the bound are freed as the parser finishes them.
const BZIP_BUFFER_POOL_MAX_BUFFERS: usize = 8;
const BZIP_BUFFER_POOL_MAX_BYTES: usize = 16 * 1024 * 1024;
/// Buffers smaller than this are not worth pooling (they would be regrown by
/// the next block anyway).
const BZIP_BUFFER_POOL_MIN_CAPACITY: usize = 64 * 1024;
/// Largest buffer the pool keeps; a pathological block that decoded to many
/// MiB is freed instead of being pinned for the life of the process.
const BZIP_BUFFER_POOL_MAX_CAPACITY: usize = 8 * 1024 * 1024;

/// Result of the Level II decoders.
pub type Result<T> = std::result::Result<T, NexradError>;

/// Why a Level II decode failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NexradError {
    /// A file could not be read.
    #[error("I/O error reading {path}: {source}")]
    Io {
        /// The path of the file.
        path: String,
        /// The I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The input is shorter than the 24-byte volume header.
    #[error("input is too short for an Archive II volume header: {actual} bytes")]
    ShortVolumeHeader {
        /// Length of the input.
        actual: usize,
    },
    /// The input does not start with an Archive II volume header (`AR2V` or
    /// `ARCHIVE2`, Table I of the Archive II ICD 2620010). Model-data
    /// (`_MDM`) files, intermediate real-time chunks and bare records have
    /// none: their messages are read with [`messages::record_bytes`] and
    /// [`messages::MessageWalker`], and their metadata with
    /// [`NexradMetadata::from_metadata_record`]. Without a header there is no
    /// site or volume time, so no [`Volume`] is built from them.
    #[error(
        "no Archive II volume header: the input starts with `{found}`, not AR2V or ARCHIVE2 (model-data _MDM files and intermediate real-time chunks have no header; read their messages with messages::MessageWalker)"
    )]
    MissingVolumeHeader {
        /// The first 8 input bytes, ASCII-escaped.
        found: String,
    },
    /// The input ends inside a structure.
    #[error("truncated {what} at offset {offset}: need {needed} bytes, have {available}")]
    Truncated {
        /// What was being read.
        what: &'static str,
        /// Where it starts.
        offset: usize,
        /// Bytes it needs.
        needed: usize,
        /// Bytes left.
        available: usize,
    },
    /// The gzip or bzip2 wrapper could not be expanded.
    #[error("unsupported or corrupt compression wrapper: {0}")]
    Compression(String),
    /// A message or record is malformed.
    #[error("invalid message at offset {offset}: {reason}")]
    InvalidMessage {
        /// Where the message starts in the (decompressed) input.
        offset: usize,
        /// What is wrong.
        reason: String,
    },
    /// The file declares more data than a documented resource limit allows
    /// (see the crate-level `# Limits` section).
    #[error("decode limit exceeded: {0}")]
    LimitExceeded(String),
}

/// The outermost compression of a Level II input, as a decoded volume's
/// `provenance.compression` names it. A gzip or whole-file bzip2 copy of an
/// LDM-compressed file is [`Self::Gzip`] or [`Self::Bzip2WholeFile`]; its
/// LDM records are decoded as the unwrapped file's are.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ArchiveCompression {
    /// A gzip file (every member decoded).
    Gzip,
    /// One bzip2 stream around the whole Archive II file.
    Bzip2WholeFile,
    /// LDM records: after the volume header, each record is a byte count
    /// and a bzip2 stream (what NOAA's archive and real-time feed carry).
    Bzip2Blocks,
    /// Uncompressed records.
    Uncompressed,
}

impl ArchiveCompression {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Gzip => "gzip",
            Self::Bzip2WholeFile => "bzip2-whole-file",
            Self::Bzip2Blocks => "bzip2-blocks",
            Self::Uncompressed => "uncompressed",
        }
    }
}

/// Message 1 / Message 31 radial status (ICD 2620002): where a radial sits in
/// its elevation cut and volume scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum RadialStatus {
    /// First radial of an elevation cut (code 0).
    StartElevation,
    /// A radial inside a cut (code 1).
    Intermediate,
    /// Last radial of a cut (code 2).
    EndElevation,
    /// First radial of the volume scan (code 3).
    StartVolume,
    /// Last radial of the volume scan (code 4).
    EndVolume,
    /// First radial of the last cut of the volume scan (code 5).
    StartElevationLastCut,
    /// Any other code, as recorded.
    Unknown(u8),
}

impl From<u8> for RadialStatus {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::StartElevation,
            1 => Self::Intermediate,
            2 => Self::EndElevation,
            3 => Self::StartVolume,
            4 => Self::EndVolume,
            5 => Self::StartElevationLastCut,
            other => Self::Unknown(other),
        }
    }
}

/// Decode a local Archive II / Level II file into the FM301 model.
pub fn read_volume_from_path(path: &Path) -> Result<Volume> {
    let mut volume = read_volume_from_bytes(&read_file(path)?)?;
    volume.provenance.source_path = Some(path.display().to_string());
    Ok(volume)
}

pub(crate) fn read_file(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|source| NexradError::Io {
        path: path.display().to_string(),
        source,
    })
}

/// Decode an Archive II / Level II byte buffer (gzip, whole-file bzip2, LDM
/// block-bzip2 or uncompressed) into the FM301 model.
pub fn read_volume_from_bytes(bytes: &[u8]) -> Result<Volume> {
    Ok(builder_from_bytes(bytes)?.finish()?.0)
}

/// Receives each message 31 body the volume decoders turn into a radial,
/// right after the radial is added, with the volume as it then stands.
/// [`metadata::read_volume_with_metadata`] uses it to read the constant
/// blocks of each sweep's first radial during the single decode pass; the
/// plain decoders pass `()`, whose empty implementation compiles away.
trait RadialObserver {
    /// A message 31 radial was decoded into ray `ray` of sweep `sweep` of
    /// `volume`.
    fn message_31(&mut self, body: &[u8], volume: &Volume, sweep: usize, ray: usize);
}

impl RadialObserver for () {
    #[inline(always)]
    fn message_31(&mut self, _body: &[u8], _volume: &Volume, _sweep: usize, _ray: usize) {}
}

pub(crate) fn builder_from_bytes(bytes: &[u8]) -> Result<VolumeBuilder> {
    builder_observed(bytes, &mut ())
}

/// [`builder_from_bytes`] with a [`RadialObserver`].
fn builder_observed(bytes: &[u8], observer: &mut impl RadialObserver) -> Result<VolumeBuilder> {
    if bytes.len() < VOLUME_HEADER_LEN {
        return Err(NexradError::ShortVolumeHeader {
            actual: bytes.len(),
        });
    }
    if !bytes.starts_with(&[0x1f, 0x8b])
        && !bytes.starts_with(b"BZh")
        && let Some(blocks) = collect_bzip_block_slices(bytes)?
    {
        return decode_bzip_blocks_pipelined(
            bytes,
            blocks,
            BlockDecode::ldm(),
            None,
            false,
            |_| Ok(()),
            observer,
        )
        .map(|outcome| outcome.builder);
    }

    if bytes.starts_with(&[0x1f, 0x8b]) {
        // Parse while inflating, through a window of about one inflate
        // chunk, instead of holding the whole expanded volume beside the
        // decoded one.
        let mut source =
            gzip::GzipRecordBytes::new(bytes, MAX_DECODED_RADAR_BYTES, "gzip radar payload");
        let parsed = parse_wrapped(&mut source, ArchiveCompression::Gzip, observer);
        return source.finish_after(parsed);
    }
    if !bytes.starts_with(b"BZh") {
        // Uncompressed records are parsed in place.
        if bytes.len() > MAX_DECODED_RADAR_BYTES {
            return Err(NexradError::Compression(format!(
                "uncompressed radar payload is {} bytes (limit {MAX_DECODED_RADAR_BYTES})",
                bytes.len()
            )));
        }
        return parse_record_stream(
            &mut SliceRecordBytes(bytes),
            ArchiveCompression::Uncompressed,
            DecodeBudget::volume(),
            observer,
        );
    }

    // Whole-file bzip2: parse while decoding block by block, instead of
    // holding the whole expanded volume beside the decoded one.
    let mut source = bzip2_prefix::Bzip2RecordBytes::new(
        bytes,
        MAX_DECODED_RADAR_BYTES,
        "whole-file bzip2 radar payload",
    );
    let parsed = parse_wrapped(&mut source, ArchiveCompression::Bzip2WholeFile, observer);
    source.finish_after(parsed)
}

/// Parse the Archive II file inside a whole-file gzip or bzip2 wrapper as
/// the wrapper expands. When that file is made of LDM records (a gzip or
/// bzip2 copy of an LDM-compressed file) its records are compressed
/// already: the wrapper is expanded whole, which holds about the size of
/// the file inside, and the records are decoded as the unwrapped file's
/// are, with `compression` naming the wrapper.
fn parse_wrapped(
    source: &mut impl RecordBytes,
    compression: ArchiveCompression,
    observer: &mut impl RadialObserver,
) -> Result<VolumeBuilder> {
    let probed = source.extend_to(LDM_PROBE_LEN)?;
    if !ldm_records_follow_header(source.get(0, probed)?) {
        return parse_record_stream(source, compression, DecodeBudget::volume(), observer);
    }
    // The source enforces MAX_DECODED_RADAR_BYTES on the expansion.
    let expanded = source.extend_to(usize::MAX)?;
    builder_from_normalized_observed(
        source.get(0, expanded)?,
        compression,
        DecodeBudget::volume(),
        observer,
    )
}

/// Decode a gzip-compressed Archive II stream into the FM301 model,
/// inflating as it parses. Every gzip member is decoded; bytes after the last
/// member that do not start another one are ignored.
pub fn read_gzip_volume_from_reader(reader: impl Read) -> Result<Volume> {
    Ok(builder_from_gzip_reader(reader)?.finish()?.0)
}

pub(crate) fn builder_from_gzip_reader(reader: impl Read) -> Result<VolumeBuilder> {
    let decoder = gzip::MultiGzReader::new(BufReader::new(reader));
    let mut decoder = ReadLimit::new(decoder, MAX_DECODED_RADAR_BYTES, "gzip radar payload");
    let mut probe = [0; LDM_PROBE_LEN];
    let probed = read_up_to(&mut decoder, &mut probe)?;
    if ldm_records_follow_header(&probe[..probed]) {
        // LDM records inside: expand the wrapper whole (see `parse_wrapped`).
        let mut expanded = probe[..probed].to_vec();
        decoder
            .read_to_end(&mut expanded)
            .map_err(|err| NexradError::Compression(err.to_string()))?;
        return builder_from_normalized(
            &expanded,
            ArchiveCompression::Gzip,
            DecodeBudget::volume(),
        );
    }
    let mut stream = (&probe[..probed]).chain(decoder);
    decode_volume_from_stream_until(&mut stream, ArchiveCompression::Gzip, None).map(|result| {
        debug_assert!(!result.stopped_at_preview);
        result.builder
    })
}

/// Read into `buffer` until it is full or the reader ends; the byte count.
fn read_up_to<R: Read>(reader: &mut R, buffer: &mut [u8]) -> Result<usize> {
    let mut read = 0;
    while read < buffer.len() {
        let count = reader
            .read(&mut buffer[read..])
            .map_err(|err| NexradError::Compression(err.to_string()))?;
        if count == 0 {
            break;
        }
        read += count;
    }
    Ok(read)
}

/// The whole expansion of a gzip file whose Archive II file inside is made
/// of LDM records, for the gzip preview decoders; `None` (after inflating
/// only its first bytes) when it is not.
fn inflate_gzip_wrapped_ldm(raw: &[u8]) -> Result<Option<Vec<u8>>> {
    let mut probe = [0; LDM_PROBE_LEN];
    let probed = read_up_to(&mut gzip::MultiGzReader::new(raw), &mut probe)?;
    if !ldm_records_follow_header(&probe[..probed]) {
        return Ok(None);
    }
    decompress_gzip_bytes(raw).map(Some)
}

/// Decode a gzip-wrapped volume, calling `on_preview` once with a sealed copy
/// of the volume as soon as its first displayable sweep completes (see
/// [`read_volume_from_bytes_with_bzip_preview`]). Other wrappers decode
/// normally without a preview.
pub fn read_gzip_volume_from_bytes_with_preview<F>(
    raw: &[u8],
    min_displayable_radials: usize,
    mut on_preview: F,
) -> Result<Volume>
where
    F: FnMut(Volume),
{
    builder_from_gzip_bytes_with_preview(raw, min_displayable_radials, |builder| {
        on_preview(builder.snapshot()?);
        Ok(())
    })?
    .finish()
    .map(|(volume, _)| volume)
}

pub(crate) fn builder_from_gzip_bytes_with_preview(
    raw: &[u8],
    min_displayable_radials: usize,
    on_preview: impl FnMut(&VolumeBuilder) -> Result<()>,
) -> Result<VolumeBuilder> {
    if raw.len() < VOLUME_HEADER_LEN {
        return Err(NexradError::ShortVolumeHeader { actual: raw.len() });
    }
    if !raw.starts_with(&[0x1f, 0x8b]) {
        return builder_from_bytes(raw);
    }
    if let Some(expanded) = inflate_gzip_wrapped_ldm(raw)? {
        let blocks = ldm_blocks_of_wrapped(&expanded)?;
        return decode_bzip_blocks_pipelined(
            &expanded,
            blocks,
            BlockDecode {
                compression: ArchiveCompression::Gzip,
                budget: DecodeBudget::volume(),
            },
            Some(min_displayable_radials),
            false,
            on_preview,
            &mut (),
        )
        .map(|outcome| outcome.builder);
    }

    let decoder = gzip::MultiGzReader::new(raw);
    let mut decoder = ReadLimit::new(decoder, MAX_DECODED_RADAR_BYTES, "gzip radar payload");
    decode_volume_from_stream(
        &mut decoder,
        ArchiveCompression::Gzip,
        Some(min_displayable_radials),
        false,
        on_preview,
    )
    .map(|result| {
        debug_assert!(!result.stopped_at_preview);
        result.builder
    })
}

/// Decode a gzip-wrapped volume only until its first displayable sweep
/// completes; `None` for other wrappers or when no sweep completes.
pub fn read_gzip_preview_from_bytes(
    raw: &[u8],
    min_displayable_radials: usize,
) -> Result<Option<Volume>> {
    match builder_gzip_preview(raw, min_displayable_radials)? {
        Some(builder) => Ok(Some(builder.finish()?.0)),
        None => Ok(None),
    }
}

pub(crate) fn builder_gzip_preview(
    raw: &[u8],
    min_displayable_radials: usize,
) -> Result<Option<VolumeBuilder>> {
    if raw.len() < VOLUME_HEADER_LEN {
        return Err(NexradError::ShortVolumeHeader { actual: raw.len() });
    }
    if !raw.starts_with(&[0x1f, 0x8b]) {
        return Ok(None);
    }
    if let Some(expanded) = inflate_gzip_wrapped_ldm(raw)? {
        let blocks = ldm_blocks_of_wrapped(&expanded)?;
        let outcome = decode_bzip_blocks_pipelined(
            &expanded,
            blocks,
            BlockDecode {
                compression: ArchiveCompression::Gzip,
                budget: DecodeBudget::volume(),
            },
            Some(min_displayable_radials),
            true,
            |_| Ok(()),
            &mut (),
        )?;
        return Ok(outcome.stopped_at_preview.then_some(outcome.builder));
    }

    let decoder = gzip::MultiGzReader::new(raw);
    let mut decoder = ReadLimit::new(decoder, MAX_DECODED_RADAR_BYTES, "gzip radar payload");
    let result = decode_volume_from_stream_until(
        &mut decoder,
        ArchiveCompression::Gzip,
        Some(min_displayable_radials),
    )?;
    Ok(result.stopped_at_preview.then_some(result.builder))
}

/// Decode a completed first displayable sweep from NEXRAD block-bzip Level II
/// bytes.
///
/// This is intended for UI preview on low-core machines: it returns `None` for
/// gzip, whole-file bzip2 or uncompressed input, and it never substitutes for
/// the final full-volume decode.
pub fn read_bzip_block_preview_from_bytes(
    raw: &[u8],
    min_displayable_radials: usize,
) -> Result<Option<Volume>> {
    match builder_bzip_block_preview(raw, min_displayable_radials)? {
        Some(builder) => Ok(Some(builder.finish()?.0)),
        None => Ok(None),
    }
}

pub(crate) fn builder_bzip_block_preview(
    raw: &[u8],
    min_displayable_radials: usize,
) -> Result<Option<VolumeBuilder>> {
    if raw.len() < VOLUME_HEADER_LEN {
        return Err(NexradError::ShortVolumeHeader { actual: raw.len() });
    }

    let Some(blocks) = collect_bzip_block_slices(raw)? else {
        return Ok(None);
    };

    let outcome = decode_bzip_blocks_pipelined(
        raw,
        blocks,
        BlockDecode::ldm(),
        Some(min_displayable_radials),
        true,
        |_| Ok(()),
        &mut (),
    )?;
    Ok(outcome.stopped_at_preview.then_some(outcome.builder))
}

/// Decode a full volume while optionally emitting an early completed
/// first-sweep preview.
///
/// For block-bzip Level II files, parsing streams behind the parallel block
/// decompression, so the preview is emitted as soon as the first displayable
/// sweep completes — without decompressing or parsing anything twice. Other
/// compression formats fall back to a normal full decode.
pub fn read_volume_from_bytes_with_bzip_preview<F>(
    raw: &[u8],
    min_displayable_radials: usize,
    mut on_preview: F,
) -> Result<Volume>
where
    F: FnMut(Volume),
{
    builder_with_bzip_preview(raw, min_displayable_radials, |builder| {
        on_preview(builder.snapshot()?);
        Ok(())
    })?
    .finish()
    .map(|(volume, _)| volume)
}

pub(crate) fn builder_with_bzip_preview(
    raw: &[u8],
    min_displayable_radials: usize,
    on_preview: impl FnMut(&VolumeBuilder) -> Result<()>,
) -> Result<VolumeBuilder> {
    if raw.len() < VOLUME_HEADER_LEN {
        return Err(NexradError::ShortVolumeHeader { actual: raw.len() });
    }

    let Some(blocks) = collect_bzip_block_slices(raw)? else {
        return builder_from_bytes(raw);
    };

    let outcome = decode_bzip_blocks_pipelined(
        raw,
        blocks,
        BlockDecode::ldm(),
        Some(min_displayable_radials),
        false,
        on_preview,
        &mut (),
    )?;
    Ok(outcome.builder)
}

/// Decompress or normalize an Archive II byte slice before Level II parsing:
/// the volume header and uncompressed records. LDM records inside a gzip or
/// whole-file bzip2 wrapper are decoded too, and the wrapper is returned.
pub fn normalize_archive_bytes(raw: &[u8]) -> Result<(Vec<u8>, ArchiveCompression)> {
    if raw.len() < VOLUME_HEADER_LEN {
        return Err(NexradError::ShortVolumeHeader { actual: raw.len() });
    }

    if raw.starts_with(&[0x1f, 0x8b]) {
        let decoded = decompress_gzip_bytes(raw)?;
        return Ok((decode_wrapped_ldm(decoded)?, ArchiveCompression::Gzip));
    }

    if raw.starts_with(b"BZh") {
        let mut decoded = Vec::new();
        decompress_bzip2_stream_into(
            raw,
            &mut decoded,
            MAX_DECODED_RADAR_BYTES,
            "whole-file bzip2 radar payload",
        )?;
        return Ok((
            decode_wrapped_ldm(decoded)?,
            ArchiveCompression::Bzip2WholeFile,
        ));
    }

    if let Some(decoded) = try_decode_bzip_blocks(raw)? {
        return Ok((decoded, ArchiveCompression::Bzip2Blocks));
    }

    Ok((
        copy_bytes_limited(raw, MAX_DECODED_RADAR_BYTES, "uncompressed radar payload")?,
        ArchiveCompression::Uncompressed,
    ))
}

/// The expansion of a whole-file wrapper with the LDM records inside it (if
/// it holds any) decoded, so the result is uncompressed records.
fn decode_wrapped_ldm(expanded: Vec<u8>) -> Result<Vec<u8>> {
    Ok(try_decode_bzip_blocks(&expanded)?.unwrap_or(expanded))
}

/// The LDM records of a wrapper's expansion that [`ldm_records_follow_header`]
/// accepted.
fn ldm_blocks_of_wrapped(expanded: &[u8]) -> Result<Vec<&[u8]>> {
    collect_bzip_block_slices(expanded)?.ok_or_else(|| {
        NexradError::Compression("LDM records inside the wrapper disappeared".to_owned())
    })
}

/// Inflate a whole in-memory gzip file (every member) into one buffer
/// presized from its ISIZE trailer. See [`gzip::inflate_gzip_members_limited`].
fn decompress_gzip_bytes(raw: &[u8]) -> Result<Vec<u8>> {
    gzip::inflate_gzip_members_limited(raw, MAX_DECODED_RADAR_BYTES, "gzip radar payload")
        .map_err(NexradError::Compression)
}

/// [`bounded_read::copy_bytes_limited`] with its message wrapped as a
/// compression error.
fn copy_bytes_limited(bytes: &[u8], limit: usize, context: &'static str) -> Result<Vec<u8>> {
    bounded_read::copy_bytes_limited(bytes, limit, context).map_err(NexradError::Compression)
}

/// Streaming expansion guard used by the preview decoders. It permits EOF
/// exactly at the limit, but probes one byte further before reporting EOF so
/// an over-limit stream cannot be mistaken for a complete volume.
struct ReadLimit<R> {
    inner: R,
    remaining: usize,
    context: &'static str,
}

impl<R> ReadLimit<R> {
    fn new(inner: R, limit: usize, context: &'static str) -> Self {
        Self {
            inner,
            remaining: limit,
            context,
        }
    }
}

impl<R: Read> Read for ReadLimit<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            let mut probe = [0u8; 1];
            return match self.inner.read(&mut probe)? {
                0 => Ok(0),
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "{} expands beyond the {}-byte limit",
                        self.context, MAX_DECODED_RADAR_BYTES
                    ),
                )),
            };
        }
        let allowed = buffer.len().min(self.remaining);
        let count = self.inner.read(&mut buffer[..allowed])?;
        self.remaining -= count;
        Ok(count)
    }
}

fn read_record_prefix<R: Read>(reader: &mut R, buffer: &mut [u8], offset: usize) -> Result<bool> {
    let mut read = 0;
    while read < buffer.len() {
        let count = reader
            .read(&mut buffer[read..])
            .map_err(|err| NexradError::Compression(err.to_string()))?;
        if count == 0 {
            if read == 0 {
                return Ok(false);
            }
            return Err(NexradError::Truncated {
                what: "record prefix",
                offset,
                needed: buffer.len(),
                available: read,
            });
        }
        read += count;
    }
    Ok(true)
}

fn read_exact_required<R: Read>(
    reader: &mut R,
    buffer: &mut [u8],
    what: &'static str,
    offset: usize,
) -> Result<()> {
    let mut read = 0;
    while read < buffer.len() {
        let count = reader
            .read(&mut buffer[read..])
            .map_err(|err| NexradError::Compression(err.to_string()))?;
        if count == 0 {
            return Err(NexradError::Truncated {
                what,
                offset,
                needed: buffer.len(),
                available: read,
            });
        }
        read += count;
    }
    Ok(())
}

fn read_exact_into_buffer<R: Read>(
    reader: &mut R,
    buffer: &mut Vec<u8>,
    len: usize,
    what: &'static str,
    offset: usize,
) -> Result<()> {
    // The buffer is reused across records without clearing: `resize` only
    // zero-fills bytes past the previous length, and every byte below `len`
    // is overwritten by the read (or the caller discards the buffer on error).
    buffer.resize(len, 0);
    read_exact_required(reader, buffer, what, offset)
}

fn skip_record_padding<R: Read>(
    reader: &mut R,
    record_len: usize,
    consumed: usize,
    record_offset: usize,
) -> Result<()> {
    let padding = record_len.saturating_sub(consumed);
    skip_exact(reader, padding, "record padding", record_offset + consumed)
}

fn skip_exact<R: Read>(
    reader: &mut R,
    mut bytes: usize,
    what: &'static str,
    offset: usize,
) -> Result<()> {
    let mut buffer = [0; 8192];
    let mut skipped = 0;
    while bytes > 0 {
        let chunk = bytes.min(buffer.len());
        let target = &mut buffer[..chunk];
        read_exact_required(reader, target, what, offset + skipped)?;
        bytes -= chunk;
        skipped += chunk;
    }
    Ok(())
}

/// Parse already-normalized Archive II bytes into the FM301 model.
///
/// `compression` is what `provenance.compression` reports. Bytes that are
/// still LDM records (the expansion of a gzip or bzip2 copy of an
/// LDM-compressed file) are decoded as LDM records.
pub fn read_normalized_volume_bytes(
    bytes: &[u8],
    compression: ArchiveCompression,
) -> Result<Volume> {
    Ok(
        builder_from_normalized(bytes, compression, DecodeBudget::volume())?
            .finish()?
            .0,
    )
}

/// [`read_normalized_volume_bytes`] with an explicit output budget.
pub(crate) fn builder_from_normalized(
    bytes: &[u8],
    compression: ArchiveCompression,
    budget: DecodeBudget,
) -> Result<VolumeBuilder> {
    builder_from_normalized_observed(bytes, compression, budget, &mut ())
}

/// [`read_normalized_volume_bytes`] with an explicit output budget and a
/// [`RadialObserver`].
fn builder_from_normalized_observed(
    bytes: &[u8],
    compression: ArchiveCompression,
    budget: DecodeBudget,
    observer: &mut impl RadialObserver,
) -> Result<VolumeBuilder> {
    if let Some(blocks) = collect_bzip_block_slices(bytes)? {
        let decode = BlockDecode {
            compression,
            budget,
        };
        return decode_bzip_blocks_pipelined(
            bytes,
            blocks,
            decode,
            None,
            false,
            |_| Ok(()),
            observer,
        )
        .map(|outcome| outcome.builder);
    }
    parse_record_stream(&mut SliceRecordBytes(bytes), compression, budget, observer)
}

/// Normalized Archive II bytes (volume header and records) as the record
/// parser reads them: by absolute offset, only near its cursor, with the
/// input's end discovered by asking for more.
pub(crate) trait RecordBytes {
    /// Make the bytes up to `end` readable and return how far the stream
    /// extends, at most `end`.
    fn extend_to(&mut self, end: usize) -> Result<usize>;
    /// Bytes `start..end`, which an earlier [`Self::extend_to`] made
    /// readable and no [`Self::release_before`] released. A range outside
    /// the readable bytes is an error, not a panic.
    fn get(&self, start: usize, end: usize) -> Result<&[u8]>;
    /// The parser reads nothing before `offset` again.
    fn release_before(&mut self, _offset: usize) {}
}

/// A whole normalized buffer.
struct SliceRecordBytes<'a>(&'a [u8]);

impl RecordBytes for SliceRecordBytes<'_> {
    #[inline]
    fn extend_to(&mut self, end: usize) -> Result<usize> {
        Ok(end.min(self.0.len()))
    }

    #[inline]
    fn get(&self, start: usize, end: usize) -> Result<&[u8]> {
        window_range(self.0, 0, start, end)
    }
}

/// Bytes `start..end` of a stream of which `window` holds `base..`, or an
/// error when the range is not inside the window. The record parser only
/// asks for bytes it made readable and did not release, so the error means
/// a parser bug, reported instead of a panic.
pub(crate) fn window_range(window: &[u8], base: usize, start: usize, end: usize) -> Result<&[u8]> {
    start
        .checked_sub(base)
        .zip(end.checked_sub(base))
        .and_then(|(from, to)| window.get(from..to))
        .ok_or_else(|| NexradError::InvalidMessage {
            offset: start,
            reason: format!(
                "record bytes {start}..{end} are outside the readable bytes {base}..{}",
                base.saturating_add(window.len())
            ),
        })
}

/// Parse normalized Archive II records (fixed 2432-byte frames and
/// variable-length messages) into a volume.
fn parse_record_stream(
    source: &mut impl RecordBytes,
    compression: ArchiveCompression,
    budget: DecodeBudget,
    observer: &mut impl RadialObserver,
) -> Result<VolumeBuilder> {
    let header_available = source.extend_to(VOLUME_HEADER_LEN)?;
    let volume_header = parse_volume_header(source.get(0, header_available)?)?;
    let mut builder = VolumeBuilder::new(
        volume_header.icao,
        volume_header.archive_version,
        volume_header.volume_time,
        compression,
        budget,
    );
    builder.record_volume_header(volume_header.date, volume_header.milliseconds);

    let mut cursor = VOLUME_HEADER_LEN;
    let mut record_index = 0usize;
    // GR2-style ".msg31" exports keep the AR2V header but carry only a few
    // metadata records before variable-framed message 31s — well before the
    // standard 134 fixed records. Detected once at the first early message
    // 31 and latched for the rest of the file.
    let mut early_variable_msg31 = false;
    loop {
        source.release_before(cursor);
        let prefix_end = cursor + CONTROL_WORD_LEN + MESSAGE_HEADER_LEN;
        if source.extend_to(prefix_end)? < prefix_end {
            break;
        }
        let header_offset = cursor + CONTROL_WORD_LEN;
        let header = parse_message_header_bytes(
            source.get(header_offset, header_offset + MESSAGE_HEADER_LEN)?,
        );

        if header.size_halfwords == 0 && record_index < 134 {
            builder.count_skipped();
            cursor = cursor.saturating_add(RECORD_BYTES);
            record_index += 1;
            continue;
        } else if header.size_halfwords == 0 {
            break;
        }

        let (message_total_len, variable_framing) = message_framing(&header);
        if message_total_len < MESSAGE_HEADER_LEN {
            return Err(NexradError::InvalidMessage {
                offset: header_offset,
                reason: "message size is smaller than message header".to_owned(),
            });
        }

        builder.count_message();
        match header.message_type {
            1 | 31 => {
                let message_end = header_offset + message_total_len;
                let available = source.extend_to(message_end)?;
                if available < message_end {
                    if builder.decoded_radials() > 0 {
                        builder.count_skipped();
                        break;
                    }
                    return Err(NexradError::Truncated {
                        what: if header.message_type == 1 {
                            "message 1 body"
                        } else {
                            "message 31 body"
                        },
                        offset: header_offset,
                        needed: message_total_len,
                        available: available.saturating_sub(header_offset),
                    });
                }
                let body = source.get(header_offset + MESSAGE_HEADER_LEN, message_end)?;
                if header.message_type == 1 {
                    parse_message_1(body, &header, &mut builder)?;
                } else {
                    let (sweep, ray) = parse_message_31(body, &header, &mut builder)?;
                    observer.message_31(body, &builder.volume, sweep, ray);
                }
            }
            _ if variable_framing => {
                // A non-radial message in variable framing (a size of 65535,
                // or message 29): kept for the model when it fits the
                // builder's cap and the input; its bytes are not read when
                // the cap refuses it.
                let body_len = message_total_len - MESSAGE_HEADER_LEN;
                if builder.keeps_variable_message(body_len) {
                    let end = header_offset.saturating_add(message_total_len);
                    if source.extend_to(end)? < end {
                        builder.count_skipped();
                    } else {
                        let message = source.get(header_offset, end)?;
                        builder.variable_message(
                            &message[..MESSAGE_HEADER_LEN],
                            &header,
                            &message[MESSAGE_HEADER_LEN..],
                        );
                    }
                } else {
                    builder.variable_message_not_kept();
                }
            }
            _ => {
                // A non-radial message in a fixed frame: the body is the rest
                // of the frame, as far as the message and the input reach.
                // A Message 5 or 7 may run past its declared size within the
                // frame (messages::vcp::fixed_frame_body_len).
                let body_offset = header_offset + MESSAGE_HEADER_LEN;
                let fixed_record_end = source.extend_to(cursor.saturating_add(RECORD_BYTES))?;
                let frame_body = source.get(body_offset.min(fixed_record_end), fixed_record_end)?;
                let body_len = messages::vcp::fixed_frame_body_len(
                    header.message_type,
                    message_total_len.saturating_sub(MESSAGE_HEADER_LEN),
                    frame_body,
                );
                builder.metadata_message(
                    source.get(header_offset, body_offset.min(fixed_record_end))?,
                    &header,
                    &frame_body[..body_len],
                    frame_body,
                );
            }
        }

        let record_len = if variable_framing {
            message_total_len + CONTROL_WORD_LEN
        } else if header.message_type != 31 {
            RECORD_BYTES
        } else if record_index >= 134 || early_variable_msg31 {
            message_total_len + CONTROL_WORD_LEN
        } else if message31_uses_variable_framing(source, cursor, message_total_len)? {
            early_variable_msg31 = true;
            message_total_len + CONTROL_WORD_LEN
        } else {
            RECORD_BYTES
        };
        cursor = cursor.saturating_add(record_len);
        record_index += 1;
    }

    Ok(builder)
}

/// Decide the framing of a message 31 seen before the standard 134 metadata
/// records. Real Archive II volumes never place message 31 that early, but
/// GR2-style ".msg31" exports (DOW/COW/RaXPol Level II twins) do, packing
/// them back to back with no fixed-record padding. Returns `true` when the
/// bytes directly after this message hold another message 31 (or the file
/// ends exactly there), which fixed 2432-byte framing cannot produce.
fn message31_uses_variable_framing(
    source: &mut impl RecordBytes,
    cursor: usize,
    message_total_len: usize,
) -> Result<bool> {
    let variable_next = cursor + CONTROL_WORD_LEN + message_total_len;
    if source.extend_to(variable_next + 1)? == variable_next {
        return Ok(true);
    }
    let header_offset = variable_next + CONTROL_WORD_LEN;
    let header_end = header_offset + MESSAGE_HEADER_LEN;
    if source.extend_to(header_end)? < header_end {
        return Ok(false);
    }
    let header = parse_message_header_bytes(source.get(header_offset, header_end)?);
    Ok(header.message_type == 31
        && usize::from(header.size_halfwords) * 2 >= MESSAGE_HEADER_LEN + MSG_31_HEADER_LEN)
}

struct StreamDecodeResult {
    builder: VolumeBuilder,
    stopped_at_preview: bool,
}

fn decode_volume_from_stream_until<R: Read>(
    reader: &mut R,
    compression: ArchiveCompression,
    preview_min_radials: Option<usize>,
) -> Result<StreamDecodeResult> {
    decode_volume_from_stream(reader, compression, preview_min_radials, true, |_| Ok(()))
}

fn decode_volume_from_stream<R: Read, F>(
    reader: &mut R,
    compression: ArchiveCompression,
    preview_min_radials: Option<usize>,
    stop_at_preview: bool,
    mut on_preview: F,
) -> Result<StreamDecodeResult>
where
    F: FnMut(&VolumeBuilder) -> Result<()>,
{
    let mut volume_header_bytes = [0; VOLUME_HEADER_LEN];
    read_exact_required(reader, &mut volume_header_bytes, "volume header", 0)?;
    let volume_header = parse_volume_header(&volume_header_bytes)?;
    let mut builder = VolumeBuilder::new(
        volume_header.icao,
        volume_header.archive_version,
        volume_header.volume_time,
        compression,
        DecodeBudget::volume(),
    );
    builder.record_volume_header(volume_header.date, volume_header.milliseconds);

    let mut cursor = VOLUME_HEADER_LEN;
    let mut record_index = 0usize;
    let mut prefix = [0; CONTROL_WORD_LEN + MESSAGE_HEADER_LEN];
    let mut body_buffer = Vec::with_capacity(RECORD_BYTES);
    let mut preview_emitted = false;
    while read_record_prefix(reader, &mut prefix, cursor)? {
        if record_index == 0 && prefix[4..7] == *b"BZh" {
            // LDM bzip2 records inside the stream (for example a gzip
            // wrapper around them): decode the records instead.
            return decode_ldm_stream(
                reader,
                &volume_header_bytes,
                &prefix,
                compression,
                preview_min_radials,
                stop_at_preview,
                on_preview,
            );
        }
        let header_offset = cursor + CONTROL_WORD_LEN;
        let header = parse_message_header_bytes(&prefix[CONTROL_WORD_LEN..]);

        if header.size_halfwords == 0 && record_index < 134 {
            builder.count_skipped();
            skip_exact(
                reader,
                RECORD_BYTES - prefix.len(),
                "empty fixed record",
                cursor + prefix.len(),
            )?;
            cursor = cursor.saturating_add(RECORD_BYTES);
            record_index += 1;
            continue;
        } else if header.size_halfwords == 0 {
            break;
        }

        let (message_total_len, variable_framing) = message_framing(&header);
        if message_total_len < MESSAGE_HEADER_LEN {
            return Err(NexradError::InvalidMessage {
                offset: header_offset,
                reason: "message size is smaller than message header".to_owned(),
            });
        }

        let record_len = if variable_framing {
            message_total_len + CONTROL_WORD_LEN
        } else if record_index < 134 || header.message_type != 31 {
            RECORD_BYTES
        } else {
            message_total_len + CONTROL_WORD_LEN
        };
        let body_len = message_total_len - MESSAGE_HEADER_LEN;
        builder.count_message();

        match header.message_type {
            1 | 31 => {
                let what = if header.message_type == 1 {
                    "message 1 body"
                } else {
                    "message 31 body"
                };
                if let Err(err) =
                    read_exact_into_buffer(reader, &mut body_buffer, body_len, what, header_offset)
                {
                    if builder.decoded_radials() > 0 {
                        builder.count_skipped();
                        break;
                    }
                    return Err(err);
                }
                if header.message_type == 1 {
                    parse_message_1(&body_buffer, &header, &mut builder)?;
                } else {
                    parse_message_31(&body_buffer, &header, &mut builder)?;
                }
                skip_record_padding(reader, record_len, prefix.len() + body_len, cursor)?;
                if let Some(min_radials) = preview_min_radials
                    && !preview_emitted
                    && builder.has_complete_displayable_sweep(min_radials)
                {
                    preview_emitted = true;
                    if stop_at_preview {
                        return Ok(StreamDecodeResult {
                            builder,
                            stopped_at_preview: true,
                        });
                    }
                    on_preview(&builder)?;
                }
            }
            _ if variable_framing => {
                if builder.keeps_variable_message(body_len) {
                    read_exact_into_buffer(
                        reader,
                        &mut body_buffer,
                        body_len,
                        "metadata message body",
                        header_offset,
                    )?;
                    builder.variable_message(
                        &prefix[CONTROL_WORD_LEN..],
                        &header,
                        &body_buffer[..body_len],
                    );
                    skip_record_padding(reader, record_len, prefix.len() + body_len, cursor)?;
                } else {
                    builder.variable_message_not_kept();
                    skip_record_padding(reader, record_len, prefix.len(), cursor)?;
                }
            }
            _ => {
                // The whole frame is read, as the padding skip would read
                // it anyway: a Message 5 or 7 may run past its declared size
                // (messages::vcp::fixed_frame_body_len), and the model keeps
                // the frame of a message it does not decode.
                let body_read_len = RECORD_BYTES.saturating_sub(prefix.len());
                read_exact_into_buffer(
                    reader,
                    &mut body_buffer,
                    body_read_len,
                    "metadata message body",
                    header_offset,
                )?;
                let used = messages::vcp::fixed_frame_body_len(
                    header.message_type,
                    body_len,
                    &body_buffer,
                );
                builder.metadata_message(
                    &prefix[CONTROL_WORD_LEN..],
                    &header,
                    &body_buffer[..used],
                    &body_buffer[..body_read_len],
                );
                skip_record_padding(reader, record_len, prefix.len() + body_read_len, cursor)?;
            }
        }

        cursor = cursor.saturating_add(record_len);
        record_index += 1;
    }

    Ok(StreamDecodeResult {
        builder,
        stopped_at_preview: false,
    })
}

/// The rest of a stream whose records are LDM bzip2 records: read it
/// (bounded by the reader's limit) and decode the records.
fn decode_ldm_stream<R: Read, F>(
    reader: &mut R,
    volume_header: &[u8],
    prefix: &[u8],
    compression: ArchiveCompression,
    preview_min_radials: Option<usize>,
    stop_at_preview: bool,
    on_preview: F,
) -> Result<StreamDecodeResult>
where
    F: FnMut(&VolumeBuilder) -> Result<()>,
{
    let mut raw = Vec::with_capacity(VOLUME_HEADER_LEN + prefix.len());
    raw.extend_from_slice(volume_header);
    raw.extend_from_slice(prefix);
    reader
        .read_to_end(&mut raw)
        .map_err(|err| NexradError::Compression(err.to_string()))?;
    let Some(blocks) = collect_bzip_block_slices(&raw)? else {
        return Err(NexradError::Compression(
            "LDM bzip2 records inside the stream do not frame".to_owned(),
        ));
    };
    decode_bzip_blocks_pipelined(
        &raw,
        blocks,
        BlockDecode {
            compression,
            budget: DecodeBudget::volume(),
        },
        preview_min_radials,
        stop_at_preview,
        on_preview,
        &mut (),
    )
    .map(|outcome| StreamDecodeResult {
        builder: outcome.builder,
        stopped_at_preview: outcome.stopped_at_preview,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VolumeHeader {
    archive_version: String,
    volume_time: DateTime<Utc>,
    icao: String,
    /// Bytes 12-15 and 16-19 as stored: modified Julian date and
    /// milliseconds past midnight.
    date: u32,
    milliseconds: u32,
}

/// The 16-byte header of a Level II message (ICD 2620002, Table II).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MessageHeader {
    /// Message size in halfwords, header included; 65535 means the size in bytes is in `segments` and `segment_number`.
    pub size_halfwords: u16,
    /// RDA redundant channel (bit 3 set: ORDA).
    pub channels: u8,
    /// Message type (1, 2, 3, 5, 13, 15, 18, 31, ...).
    pub message_type: u8,
    /// Message sequence number.
    pub sequence_id: u16,
    /// Date: days since 1 January 1970, 1 January 1970 being day 1.
    pub date: u16,
    /// Time: milliseconds after midnight UTC.
    pub milliseconds: u32,
    /// Number of segments of the message.
    pub segments: u16,
    /// Number of this segment, from 1.
    pub segment_number: u16,
}

/// The data header of a Message 31 radial (ICD 2620002, Table XVII-A).
#[derive(Clone, Debug, PartialEq)]
pub struct Message31Header {
    /// Bytes 0-3: radar identifier as recorded. It can be blank (KVWX
    /// 2008-04-15 records four spaces); see [`Self::radar_identifier_or`].
    pub radar_identifier: [u8; 4],
    /// Collection time: milliseconds after midnight UTC.
    pub collect_ms: u32,
    /// Collection date: days since 1 January 1970, 1 January 1970 being day 1.
    pub collect_date: u16,
    /// Radial number within the elevation cut, from 1.
    pub azimuth_number: u16,
    /// Azimuth angle, degrees.
    pub azimuth_angle: f32,
    /// Radial length in bytes, data header included.
    pub radial_length: u16,
    /// Azimuth spacing code (1: 0.5 degrees, 2: 1 degree).
    pub azimuth_resolution: u8,
    /// Where the radial sits in its cut and volume.
    pub radial_status: RadialStatus,
    /// Elevation number within the volume coverage pattern, from 1.
    pub elevation_number: u8,
    /// Sector number within the cut.
    pub cut_sector: u8,
    /// Elevation angle, degrees.
    pub elevation_angle: f32,
    /// Offsets of the data blocks from the start of the data header, in block order (0 for absent blocks).
    pub block_pointers: [usize; 10],
}

/// Outcome of decompressing one LDM block: its bytes, taken once by the
/// parser and then recycled, or the error message.
type BlockResult = std::result::Result<Mutex<Option<Vec<u8>>>, String>;

/// Process-wide pool of decoded LDM block buffers. A block decoded into a
/// recycled buffer neither allocates nor page-faults; without the pool every
/// volume allocated (and the allocator returned to the OS) ~100 fresh ~1 MiB
/// block buffers. Bounded by `BZIP_BUFFER_POOL_MAX_BUFFERS` and
/// `BZIP_BUFFER_POOL_MAX_BYTES`.
struct BzipBufferPool {
    buffers: Vec<Vec<u8>>,
    retained_bytes: usize,
}

static BZIP_BUFFER_POOL: Mutex<BzipBufferPool> = Mutex::new(BzipBufferPool {
    buffers: Vec::new(),
    retained_bytes: 0,
});

/// An empty buffer for one decoded block, recycled when the pool has one.
fn take_bzip_buffer() -> Vec<u8> {
    let mut pool = BZIP_BUFFER_POOL
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    match pool.buffers.pop() {
        Some(mut buffer) => {
            pool.retained_bytes = pool.retained_bytes.saturating_sub(buffer.capacity());
            buffer.clear();
            buffer
        }
        None => Vec::new(),
    }
}

/// Return a decoded block buffer to the pool, or free it when the pool is
/// full or the buffer is outside the pooled capacity range.
fn recycle_bzip_buffer(mut buffer: Vec<u8>) {
    let capacity = buffer.capacity();
    if !(BZIP_BUFFER_POOL_MIN_CAPACITY..=BZIP_BUFFER_POOL_MAX_CAPACITY).contains(&capacity) {
        return;
    }
    buffer.clear();
    let mut pool = BZIP_BUFFER_POOL
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if pool.buffers.len() >= BZIP_BUFFER_POOL_MAX_BUFFERS
        || pool.retained_bytes.saturating_add(capacity) > BZIP_BUFFER_POOL_MAX_BYTES
    {
        return;
    }
    pool.retained_bytes += capacity;
    pool.buffers.push(buffer);
}

/// Slot store connecting parallel LDM-block decompression workers to the
/// in-order streaming parser.
///
/// Indices are claimed in parse order through `next_claim`, so each slot is
/// filled by exactly one thread. A slot is a `OnceLock` that is set once with
/// the block's bytes (or error) and never written again. The parser takes
/// the bytes out of a published slot exactly once, parses past them, and
/// hands the buffer back to the buffer pool for a later block.
struct BlockSlots<'a> {
    compressed: Vec<&'a [u8]>,
    slots: Box<[OnceLock<BlockResult>]>,
    next_claim: AtomicUsize,
    decoded_bytes: AtomicUsize,
    canceled: AtomicBool,
    wakeup: Mutex<()>,
    published: Condvar,
}

impl<'a> BlockSlots<'a> {
    fn new(compressed: Vec<&'a [u8]>) -> Self {
        let len = compressed.len();
        Self {
            compressed,
            slots: (0..len).map(|_| OnceLock::new()).collect(),
            next_claim: AtomicUsize::new(0),
            decoded_bytes: AtomicUsize::new(0),
            canceled: AtomicBool::new(false),
            wakeup: Mutex::new(()),
            published: Condvar::new(),
        }
    }

    fn len(&self) -> usize {
        self.compressed.len()
    }

    fn cancel(&self) {
        self.canceled.store(true, Ordering::Relaxed);
    }

    fn run_worker(&self) {
        while !self.canceled.load(Ordering::Relaxed) {
            let index = self.claim();
            if index >= self.len() {
                break;
            }
            self.decompress_claimed(index);
        }
    }

    /// Claim the next index in parse order (with `paired-bzip2`, the next
    /// two indices; the claimant decodes both).
    fn claim(&self) -> usize {
        let step = if cfg!(feature = "paired-bzip2") { 2 } else { 1 };
        self.next_claim.fetch_add(step, Ordering::Relaxed)
    }

    #[cfg(not(feature = "paired-bzip2"))]
    fn decompress_claimed(&self, index: usize) {
        let mut decoded = take_bzip_buffer();
        match decompress_bzip_block_into(self.compressed[index], &mut decoded) {
            Ok(()) => self.publish(index, Ok(decoded)),
            Err(err) => {
                recycle_bzip_buffer(decoded);
                self.publish(index, Err(err.to_string()));
            }
        }
    }

    #[cfg(feature = "paired-bzip2")]
    fn decompress_claimed(&self, index: usize) {
        let mut first = take_bzip_buffer();
        let Some(&second_compressed) = self.compressed.get(index + 1) else {
            match decompress_bzip_block_into(self.compressed[index], &mut first) {
                Ok(()) => self.publish(index, Ok(first)),
                Err(err) => {
                    recycle_bzip_buffer(first);
                    self.publish(index, Err(err.to_string()));
                }
            }
            return;
        };
        let mut second = take_bzip_buffer();
        let (first_result, second_result) = decompress_bzip_block_pair_into(
            self.compressed[index],
            &mut first,
            second_compressed,
            &mut second,
        );
        for (index, result, buffer) in [
            (index, first_result, first),
            (index + 1, second_result, second),
        ] {
            match result {
                Ok(()) => self.publish(index, Ok(buffer)),
                Err(err) => {
                    recycle_bzip_buffer(buffer);
                    self.publish(index, Err(err.to_string()));
                }
            }
        }
    }

    /// Publish a block's decoded bytes (after charging them to the aggregate
    /// budget) or its error, then wake the parser.
    fn publish(&self, index: usize, decoded: std::result::Result<Vec<u8>, String>) {
        let result = match decoded {
            Ok(decoded) => {
                if reserve_atomic_budget(
                    &self.decoded_bytes,
                    decoded.len(),
                    MAX_DECODED_RADAR_BYTES,
                ) {
                    Ok(Mutex::new(Some(decoded)))
                } else {
                    recycle_bzip_buffer(decoded);
                    self.cancel();
                    Err(format!(
                        "block-bzip radar payload expands beyond the {MAX_DECODED_RADAR_BYTES}-byte aggregate limit"
                    ))
                }
            }
            Err(message) => Err(message),
        };
        // `index` was claimed exactly once via `next_claim`, so the slot is
        // still empty; `set` cannot fail here and would never overwrite.
        let _ = self.slots[index].set(result);
        // Take the wakeup lock so a parser that has checked the slot but not
        // yet parked cannot miss this notification. The mutex guards no data,
        // so a poisoned lock is still usable.
        drop(self.wakeup.lock().unwrap_or_else(PoisonError::into_inner));
        self.published.notify_all();
    }

    /// Block until the decompressed contents of `index` are available, then
    /// take ownership of them. Each index is taken at most once, in parse
    /// order, by the single parsing cursor.
    ///
    /// The caller participates in decompression while it waits (claims advance
    /// in parse order), so the pipeline makes progress even when no rayon
    /// worker ever runs — e.g. on a single-threaded pool.
    fn take_block(&self, index: usize) -> Result<Vec<u8>> {
        loop {
            if let Some(result) = self.slots[index].get() {
                return match result {
                    Ok(cell) => cell
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .take()
                        .ok_or_else(|| {
                            NexradError::Compression("bzip2 block was already consumed".to_owned())
                        }),
                    Err(message) => Err(NexradError::Compression(message.clone())),
                };
            }
            let claimed = self.claim();
            if claimed < self.len() {
                self.decompress_claimed(claimed);
                continue;
            }
            // Everything is claimed, so `index` is in flight on another
            // thread; park until the next publish.
            let mut guard = self.wakeup.lock().unwrap_or_else(PoisonError::into_inner);
            while self.slots[index].get().is_none() {
                guard = self
                    .published
                    .wait(guard)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        }
    }
}

impl Drop for BlockSlots<'_> {
    fn drop(&mut self) {
        // Blocks decoded ahead of an early return (preview stop, error) go
        // back to the pool instead of being freed.
        for slot in self.slots.iter_mut() {
            if let Some(Ok(cell)) = slot.get_mut()
                && let Some(buffer) = cell
                    .get_mut()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take()
            {
                recycle_bzip_buffer(buffer);
            }
        }
    }
}

/// The chunk a [`BzipBlockCursor`] is reading.
enum CursorChunk<'a> {
    /// The borrowed 24-byte volume header (chunk 0).
    Header(&'a [u8]),
    /// A decoded block, owned while it is parsed and recycled on advance.
    Block(Vec<u8>),
    /// Past the last block.
    End,
}

/// Bytes of the loaded chunk; empty when nothing is loaded or past the end.
///
/// A free function over the `current` field (rather than a `&self` method) so
/// callers can keep the returned slice while updating the cursor offsets.
fn cursor_chunk_bytes<'c>(current: &'c Option<CursorChunk<'_>>) -> &'c [u8] {
    match current {
        Some(CursorChunk::Header(bytes)) => bytes,
        Some(CursorChunk::Block(buffer)) => buffer.as_slice(),
        Some(CursorChunk::End) | None => &[],
    }
}

struct BzipBlockCursor<'a> {
    volume_header: &'a [u8],
    blocks: &'a BlockSlots<'a>,
    chunk_index: usize,
    chunk_offset: usize,
    absolute_offset: usize,
    /// `None` until `chunk_index` has been loaded.
    current: Option<CursorChunk<'a>>,
}

impl<'a> BzipBlockCursor<'a> {
    fn new(volume_header: &'a [u8], blocks: &'a BlockSlots<'a>) -> Self {
        Self {
            volume_header,
            blocks,
            chunk_index: 0,
            chunk_offset: 0,
            absolute_offset: 0,
            current: None,
        }
    }

    /// Load `chunk_index` if needed. Returns `false` past the last block.
    fn load_current(&mut self) -> Result<bool> {
        if self.current.is_none() {
            let chunk = match self.chunk_index {
                0 => CursorChunk::Header(self.volume_header),
                index if index - 1 < self.blocks.len() => {
                    CursorChunk::Block(self.blocks.take_block(index - 1)?)
                }
                _ => CursorChunk::End,
            };
            self.current = Some(chunk);
        }
        Ok(!matches!(self.current, Some(CursorChunk::End)))
    }

    fn advance_chunk(&mut self) {
        if let Some(CursorChunk::Block(buffer)) = self.current.take() {
            recycle_bzip_buffer(buffer);
        }
        self.chunk_index += 1;
        self.chunk_offset = 0;
    }

    /// Advance past exhausted chunks. Returns `true` when a chunk with unread
    /// bytes is loaded, `false` at the end of the block stream.
    fn skip_empty_chunks(&mut self) -> Result<bool> {
        while self.load_current()? {
            if self.chunk_offset < cursor_chunk_bytes(&self.current).len() {
                return Ok(true);
            }
            self.advance_chunk();
        }
        Ok(false)
    }

    fn read_exact_into(
        &mut self,
        mut output: &mut [u8],
        what: &'static str,
        offset: usize,
    ) -> Result<()> {
        let mut written = 0;
        while !output.is_empty() {
            if !self.skip_empty_chunks()? {
                return Err(NexradError::Truncated {
                    what,
                    offset,
                    needed: written + output.len(),
                    available: written,
                });
            }
            let available = &cursor_chunk_bytes(&self.current)[self.chunk_offset..];
            let count = available.len().min(output.len());
            output[..count].copy_from_slice(&available[..count]);
            self.chunk_offset += count;
            self.absolute_offset += count;
            let (_, rest) = output.split_at_mut(count);
            output = rest;
            written += count;
        }
        Ok(())
    }

    fn read_optional_prefix(&mut self, output: &mut [u8], offset: usize) -> Result<bool> {
        // Fast path: the whole prefix lies inside the loaded chunk (nearly
        // every record), so skip the chunk bookkeeping.
        let start = self.chunk_offset;
        if !output.is_empty()
            && let Some(bytes) = cursor_chunk_bytes(&self.current).get(start..start + output.len())
        {
            output.copy_from_slice(bytes);
            self.chunk_offset += output.len();
            self.absolute_offset += output.len();
            return Ok(true);
        }
        if !self.skip_empty_chunks()? {
            return Ok(false);
        }
        self.read_exact_into(output, "record prefix", offset)?;
        Ok(true)
    }

    fn read_slice_or_copy<'b>(
        &'b mut self,
        scratch: &'b mut Vec<u8>,
        len: usize,
        what: &'static str,
        offset: usize,
    ) -> Result<&'b [u8]> {
        scratch.clear();
        if len == 0 {
            return Ok(&[]);
        }
        if !self.skip_empty_chunks()? {
            return Err(NexradError::Truncated {
                what,
                offset,
                needed: len,
                available: 0,
            });
        }
        let start = self.chunk_offset;
        if start + len <= cursor_chunk_bytes(&self.current).len() {
            self.chunk_offset += len;
            self.absolute_offset += len;
            return Ok(&cursor_chunk_bytes(&self.current)[start..start + len]);
        }

        scratch.reserve_exact(len);
        let mut remaining = len;
        while remaining > 0 {
            if !self.skip_empty_chunks()? {
                return Err(NexradError::Truncated {
                    what,
                    offset,
                    needed: len,
                    available: scratch.len(),
                });
            }
            let available = &cursor_chunk_bytes(&self.current)[self.chunk_offset..];
            let count = available.len().min(remaining);
            scratch.extend_from_slice(&available[..count]);
            self.chunk_offset += count;
            self.absolute_offset += count;
            remaining -= count;
        }
        Ok(scratch.as_slice())
    }

    fn skip_exact(&mut self, len: usize, what: &'static str, offset: usize) -> Result<()> {
        // Fast path: the skip ends inside the loaded chunk.
        if self.chunk_offset + len <= cursor_chunk_bytes(&self.current).len() {
            self.chunk_offset += len;
            self.absolute_offset += len;
            return Ok(());
        }
        let mut skipped = 0;
        while skipped < len {
            if !self.skip_empty_chunks()? {
                return Err(NexradError::Truncated {
                    what,
                    offset,
                    needed: len,
                    available: skipped,
                });
            }
            let count =
                (len - skipped).min(cursor_chunk_bytes(&self.current).len() - self.chunk_offset);
            self.chunk_offset += count;
            self.absolute_offset += count;
            skipped += count;
        }
        Ok(())
    }
}

impl Drop for BzipBlockCursor<'_> {
    fn drop(&mut self) {
        if let Some(CursorChunk::Block(buffer)) = self.current.take() {
            recycle_bzip_buffer(buffer);
        }
    }
}

struct BlockParseOutcome {
    builder: VolumeBuilder,
    stopped_at_preview: bool,
}

/// How a block-bzip volume is decoded: the wrapper its provenance names
/// ([`ArchiveCompression::Bzip2Blocks`], or the whole-file wrapper around
/// the LDM records) and its output budget.
#[derive(Clone, Copy)]
struct BlockDecode {
    compression: ArchiveCompression,
    budget: DecodeBudget,
}

impl BlockDecode {
    /// An LDM-compressed file as it is, with the volume budget.
    fn ldm() -> Self {
        Self {
            compression: ArchiveCompression::Bzip2Blocks,
            budget: DecodeBudget::volume(),
        }
    }
}

fn decode_bzip_blocks_pipelined(
    raw: &[u8],
    blocks: Vec<&[u8]>,
    decode: BlockDecode,
    min_displayable_radials: Option<usize>,
    stop_at_preview: bool,
    on_preview: impl FnMut(&VolumeBuilder) -> Result<()>,
    observer: &mut impl RadialObserver,
) -> Result<BlockParseOutcome> {
    let slots = BlockSlots::new(blocks);
    rayon::in_place_scope(|scope| {
        // Leave one hardware thread for the parsing thread below; it also
        // steals decompression work whenever it would otherwise wait. A
        // one-thread pool spawns no worker at all: the parser decodes every
        // block itself instead of time-slicing against a worker on one core.
        let workers = rayon::current_num_threads()
            .saturating_sub(1)
            .min(slots.len());
        for _ in 0..workers {
            scope.spawn(|_| slots.run_worker());
        }
        let outcome = parse_bzip_block_volume(
            &raw[..VOLUME_HEADER_LEN],
            &slots,
            decode,
            min_displayable_radials,
            stop_at_preview,
            on_preview,
            observer,
        );
        // Stop idle claims if the parse returned early (preview-only or error).
        slots.cancel();
        outcome
    })
}

fn parse_bzip_block_volume(
    volume_header: &[u8],
    blocks: &BlockSlots<'_>,
    decode: BlockDecode,
    min_displayable_radials: Option<usize>,
    stop_at_preview: bool,
    mut on_preview: impl FnMut(&VolumeBuilder) -> Result<()>,
    observer: &mut impl RadialObserver,
) -> Result<BlockParseOutcome> {
    let mut preview_pending = min_displayable_radials;
    let mut cursor_reader = BzipBlockCursor::new(volume_header, blocks);
    let mut volume_header_buffer = Vec::new();
    let volume_header_bytes = cursor_reader.read_slice_or_copy(
        &mut volume_header_buffer,
        VOLUME_HEADER_LEN,
        "volume header",
        0,
    )?;
    let volume_header = parse_volume_header(volume_header_bytes)?;
    let mut builder = VolumeBuilder::new(
        volume_header.icao,
        volume_header.archive_version,
        volume_header.volume_time,
        decode.compression,
        decode.budget,
    );
    builder.record_volume_header(volume_header.date, volume_header.milliseconds);

    let mut cursor = VOLUME_HEADER_LEN;
    let mut record_index = 0usize;
    let mut prefix = [0; CONTROL_WORD_LEN + MESSAGE_HEADER_LEN];
    let mut body_buffer = Vec::with_capacity(RECORD_BYTES);
    loop {
        match cursor_reader.read_optional_prefix(&mut prefix, cursor) {
            Ok(true) => {}
            Ok(false) => break,
            // A damaged block after at least one decoded radial degrades to a
            // partial volume, mirroring the message-31 body handling below.
            Err(err) => {
                if builder.decoded_radials() > 0 {
                    builder.count_skipped();
                    break;
                }
                return Err(err);
            }
        }
        let header_offset = cursor + CONTROL_WORD_LEN;
        let header = parse_message_header_bytes(&prefix[CONTROL_WORD_LEN..]);
        // Chunk 0 is the volume header and chunk 1 the first LDM record (the
        // metadata record); a message in a later record is radial data.
        let in_data_record = cursor_reader.chunk_index >= 2;

        if header.size_halfwords == 0 && record_index < 134 {
            builder.count_skipped();
            cursor_reader.skip_exact(
                RECORD_BYTES - prefix.len(),
                "empty fixed record",
                cursor + prefix.len(),
            )?;
            cursor = cursor.saturating_add(RECORD_BYTES);
            record_index += 1;
            continue;
        } else if header.size_halfwords == 0 {
            break;
        }

        let (message_total_len, variable_framing) = message_framing(&header);
        if message_total_len < MESSAGE_HEADER_LEN {
            return Err(NexradError::InvalidMessage {
                offset: header_offset,
                reason: "message size is smaller than message header".to_owned(),
            });
        }

        // Message 31 radials are variable length except inside the 134
        // fixed frames of the metadata record. A first LDM record with fewer
        // frames (some converted files carry only Messages 18 and 5 there) ends the
        // fixed frames early: radials of later records are variable length.
        let record_len = if variable_framing {
            message_total_len + CONTROL_WORD_LEN
        } else if header.message_type != 31 || (record_index < 134 && !in_data_record) {
            RECORD_BYTES
        } else {
            message_total_len + CONTROL_WORD_LEN
        };
        let body_len = message_total_len - MESSAGE_HEADER_LEN;
        builder.count_message();

        match header.message_type {
            1 | 31 => {
                let what = if header.message_type == 1 {
                    "message 1 body"
                } else {
                    "message 31 body"
                };
                let body = match cursor_reader.read_slice_or_copy(
                    &mut body_buffer,
                    body_len,
                    what,
                    header_offset,
                ) {
                    Ok(body) => body,
                    Err(err) => {
                        if builder.decoded_radials() > 0 {
                            builder.count_skipped();
                            break;
                        }
                        return Err(err);
                    }
                };
                if header.message_type == 1 {
                    parse_message_1(body, &header, &mut builder)?;
                } else {
                    let (sweep, ray) = parse_message_31(body, &header, &mut builder)?;
                    observer.message_31(body, &builder.volume, sweep, ray);
                }
                if let Some(min_radials) = preview_pending
                    && builder.has_complete_displayable_sweep(min_radials)
                {
                    preview_pending = None;
                    if stop_at_preview {
                        return Ok(BlockParseOutcome {
                            builder,
                            stopped_at_preview: true,
                        });
                    }
                    on_preview(&builder)?;
                }
                cursor_reader.skip_exact(
                    record_len.saturating_sub(prefix.len() + body_len),
                    "record padding",
                    cursor + prefix.len() + body_len,
                )?;
            }
            _ if variable_framing => {
                let kept = if builder.keeps_variable_message(body_len) {
                    let body = cursor_reader.read_slice_or_copy(
                        &mut body_buffer,
                        body_len,
                        "metadata message body",
                        header_offset,
                    )?;
                    builder.variable_message(&prefix[CONTROL_WORD_LEN..], &header, body);
                    body_len
                } else {
                    builder.variable_message_not_kept();
                    0
                };
                cursor_reader.skip_exact(
                    record_len.saturating_sub(prefix.len() + kept),
                    "record padding",
                    cursor + prefix.len() + kept,
                )?;
            }
            _ => {
                // The whole frame is read, as the padding skip would read
                // it anyway: a Message 5 or 7 may run past its declared size
                // (messages::vcp::fixed_frame_body_len), and the model keeps
                // the frame of a message it does not decode.
                let body_read_len = RECORD_BYTES.saturating_sub(prefix.len());
                let body = cursor_reader.read_slice_or_copy(
                    &mut body_buffer,
                    body_read_len,
                    "metadata message body",
                    header_offset,
                )?;
                let used = messages::vcp::fixed_frame_body_len(header.message_type, body_len, body);
                builder.metadata_message(&prefix[CONTROL_WORD_LEN..], &header, &body[..used], body);
                cursor_reader.skip_exact(
                    record_len.saturating_sub(prefix.len() + body_read_len),
                    "record padding",
                    cursor + prefix.len() + body_read_len,
                )?;
            }
        }

        cursor = cursor.saturating_add(record_len);
        record_index += 1;
    }

    Ok(BlockParseOutcome {
        builder,
        stopped_at_preview: false,
    })
}

fn try_decode_bzip_blocks(raw: &[u8]) -> Result<Option<Vec<u8>>> {
    let Some(decoded_blocks) = try_decompress_bzip_blocks(raw)? else {
        return Ok(None);
    };

    let decoded_len = decoded_blocks.iter().try_fold(0usize, |total, block| {
        total.checked_add(block.len()).filter(|sum| *sum <= MAX_DECODED_RADAR_BYTES)
    }).ok_or_else(|| {
        NexradError::Compression(format!(
            "block-bzip radar payload expands beyond the {MAX_DECODED_RADAR_BYTES}-byte aggregate limit"
        ))
    })?;
    let output_len = VOLUME_HEADER_LEN
        .checked_add(decoded_len)
        .ok_or_else(|| NexradError::Compression("block-bzip output size overflow".to_owned()))?;
    let mut output = Vec::new();
    output.try_reserve_exact(output_len).map_err(|err| {
        NexradError::Compression(format!("cannot reserve block-bzip output: {err}"))
    })?;
    output.extend_from_slice(&raw[..VOLUME_HEADER_LEN]);
    for block in decoded_blocks {
        output.extend_from_slice(&block);
        recycle_bzip_buffer(block);
    }

    Ok(Some(output))
}

fn try_decompress_bzip_blocks(raw: &[u8]) -> Result<Option<Vec<Vec<u8>>>> {
    let Some(blocks) = collect_bzip_block_slices(raw)? else {
        return Ok(None);
    };

    let decoded_bytes = AtomicUsize::new(0);
    let decoded_blocks = blocks
        .par_iter()
        .map(|compressed| {
            let mut decoded = take_bzip_buffer();
            if let Err(err) = decompress_bzip_block_into(compressed, &mut decoded) {
                recycle_bzip_buffer(decoded);
                return Err(err);
            }
            if !reserve_atomic_budget(
                &decoded_bytes,
                decoded.len(),
                MAX_DECODED_RADAR_BYTES,
            ) {
                recycle_bzip_buffer(decoded);
                return Err(NexradError::Compression(format!(
                    "block-bzip radar payload expands beyond the {MAX_DECODED_RADAR_BYTES}-byte aggregate limit"
                )));
            }
            Ok(decoded)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Some(decoded_blocks))
}

/// The LDM records of an Archive II file held whole in memory: after the
/// volume header, each record is a big-endian `i32` byte count (negative
/// for the last record) and a bzip2 stream. `None` when the bytes after
/// the volume header do not start with such a record (uncompressed
/// records, or too few bytes to tell).
///
/// Once the first record is an LDM record the file is read as LDM records
/// to its end, never as uncompressed messages: a zero count, or a lone `-1`
/// in the last four bytes, ends the records; a record that is not a bzip2
/// stream is kept, and fails to decode as a damaged record does; a record
/// that the end of the input cuts short is kept as far as it goes and fails
/// the same way, so a truncated file decodes to the records before it
/// (the first record cut short is a [`NexradError::Truncated`] error).
fn collect_bzip_block_slices(raw: &[u8]) -> Result<Option<Vec<&[u8]>>> {
    if !ldm_records_follow_header(raw) {
        return Ok(None);
    }

    let mut cursor = VOLUME_HEADER_LEN;
    let mut blocks = Vec::new();

    while cursor + 4 <= raw.len() {
        let signed_block_size = i32_at(raw, cursor)?;
        if signed_block_size == 0 || (signed_block_size == -1 && cursor + 4 == raw.len()) {
            break;
        }

        let control_offset = cursor;
        cursor += 4;
        let is_last_block = signed_block_size < 0;
        let block_size = usize::try_from(signed_block_size.unsigned_abs())
            .map_err(|_| NexradError::Compression("bzip2 block size overflow".to_owned()))?;
        let available = raw.len() - cursor;
        if block_size > available && blocks.is_empty() {
            return Err(NexradError::Truncated {
                what: "LDM compressed record",
                offset: control_offset,
                needed: block_size,
                available,
            });
        }
        let end = cursor + block_size.min(available);

        blocks.push(&raw[cursor..end]);
        if blocks.len() > MAX_BZIP_BLOCKS {
            return Err(NexradError::Compression(format!(
                "block-bzip volume contains more than {MAX_BZIP_BLOCKS} blocks"
            )));
        }
        cursor = end;
        if is_last_block {
            break;
        }
    }

    if blocks.is_empty() {
        return Err(NexradError::Compression(
            "LDM-compressed volume has no record: its first byte count is zero".to_owned(),
        ));
    }

    Ok(Some(blocks))
}

/// True when the bytes after the 24-byte volume header start with an LDM
/// record: a four-byte count, then a bzip2 stream header. Uncompressed
/// records have the 12-byte control word there, which is never `BZh`.
pub(crate) fn ldm_records_follow_header(bytes: &[u8]) -> bool {
    bytes
        .get(VOLUME_HEADER_LEN..)
        .is_some_and(messages::starts_with_ldm_record)
}

/// Bytes a decoder must have expanded to tell LDM records from
/// uncompressed ones ([`ldm_records_follow_header`]).
const LDM_PROBE_LEN: usize = VOLUME_HEADER_LEN + 4 + 3;

thread_local! {
    /// Per-thread bzip2 decoder: its block work buffers (about 7 MiB of
    /// address space, resident as far as the largest block touched them) are
    /// reused across every LDM record and whole-file stream this thread
    /// decodes.
    static BZIP2_DECODER: RefCell<recast_radar_bzip2::Decoder> =
        RefCell::new(recast_radar_bzip2::Decoder::new());
}

/// Wrap a decoder error for `context`. The output limit keeps the wording of
/// `bounded_read::read_to_end_limited`, which this path used before.
fn bzip2_error(err: recast_radar_bzip2::Error, limit: usize, context: &'static str) -> NexradError {
    NexradError::Compression(match err {
        recast_radar_bzip2::Error::OutputLimit => {
            format!("{context} expands beyond the {limit}-byte limit")
        }
        other => format!("{context}: {other}"),
    })
}

/// Decode one bzip2 stream (`BZh1`..`BZh9` header through the end-of-stream
/// marker) with this thread's decoder and append it to `output`. Block CRCs
/// and the combined stream CRC are verified; bytes after the end-of-stream
/// marker are ignored. At most `limit` bytes are appended: the limit is
/// checked against each block's exact decoded size before that block's
/// output is allocated, and on any error `output` is back at its original
/// length.
fn decompress_bzip2_stream_into(
    compressed: &[u8],
    output: &mut Vec<u8>,
    limit: usize,
    context: &'static str,
) -> Result<()> {
    BZIP2_DECODER.with(|decoder| {
        let mut decoder = decoder.borrow_mut();
        decoder.set_max_output(limit);
        decoder
            .decode_stream_into(compressed, output)
            .map_err(|err| bzip2_error(err, limit, context))
    })
}

/// Decode one LDM block-bzip record (a complete bzip2 stream) into `output`,
/// replacing its contents but keeping its capacity.
fn decompress_bzip_block_into(compressed: &[u8], output: &mut Vec<u8>) -> Result<()> {
    output.clear();
    decompress_bzip2_stream_into(
        compressed,
        output,
        MAX_BZIP_BLOCK_DECODED_BYTES,
        "block-bzip chunk",
    )
}

/// Two LDM records decoded in lockstep on this thread, each into its own
/// buffer (contents replaced, capacity kept). The results are independent:
/// a corrupt record does not affect the other.
#[cfg(feature = "paired-bzip2")]
fn decompress_bzip_block_pair_into(
    first: &[u8],
    first_output: &mut Vec<u8>,
    second: &[u8],
    second_output: &mut Vec<u8>,
) -> (Result<()>, Result<()>) {
    first_output.clear();
    second_output.clear();
    BZIP2_DECODER.with(|decoder| {
        let mut decoder = decoder.borrow_mut();
        decoder.set_max_output(MAX_BZIP_BLOCK_DECODED_BYTES);
        let (first_result, second_result) =
            decoder.decode_two_into(first, first_output, second, second_output);
        let wrap = |result: std::result::Result<(), recast_radar_bzip2::Error>| {
            result.map_err(|err| bzip2_error(err, MAX_BZIP_BLOCK_DECODED_BYTES, "block-bzip chunk"))
        };
        (wrap(first_result), wrap(second_result))
    })
}

/// Add `additional` to `total` unless the sum would pass `limit`; `false`
/// (and `total` unchanged) when it would.
///
/// A compare-exchange loop, which is what `AtomicUsize::fetch_update` does;
/// nightly deprecates `fetch_update` for `try_update`, which is newer than
/// the minimum Rust version.
fn reserve_atomic_budget(total: &AtomicUsize, additional: usize, limit: usize) -> bool {
    let mut current = total.load(Ordering::Acquire);
    loop {
        let Some(next) = current
            .checked_add(additional)
            .filter(|next| *next <= limit)
        else {
            return false;
        };
        match total.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(actual) => current = actual,
        }
    }
}

/// True when `bytes` start with an Archive II volume header tape name:
/// `AR2V` (Build 5 on, 2004) or `ARCHIVE2` (1991-2003).
pub(crate) fn starts_with_volume_header(bytes: &[u8]) -> bool {
    bytes.starts_with(b"AR2V") || bytes.starts_with(b"ARCHIVE2")
}

/// Parse the 24-byte volume header. Bytes that do not start with `AR2V` or
/// `ARCHIVE2` are [`NexradError::MissingVolumeHeader`]: the volume decoders
/// never treat headerless bytes (a model-data file, an intermediate
/// real-time chunk, compressed data) as a header followed by records.
fn parse_volume_header(bytes: &[u8]) -> Result<VolumeHeader> {
    require_len(bytes, 0, VOLUME_HEADER_LEN, "volume header")?;
    if !starts_with_volume_header(bytes) {
        return Err(NexradError::MissingVolumeHeader {
            found: bytes[..8].escape_ascii().to_string(),
        });
    }
    let tape = ascii_trim(&bytes[0..9]);
    let extension = ascii_trim(&bytes[9..12]);
    let date = u32_at(bytes, 12)?;
    let milliseconds = u32_at(bytes, 16)?;
    let icao = ascii_trim(&bytes[20..24]);

    Ok(VolumeHeader {
        archive_version: format!("{tape}{extension}"),
        volume_time: nexrad_date_ms_to_datetime(date, milliseconds),
        icao,
        date,
        milliseconds,
    })
}

/// Framing of one message in the volume decoders: the message length from
/// its header (Table II: the size halfwords, or the byte count in the segment
/// fields when the size is the 0xFFFF sentinel, notes 6 and 7) and whether
/// the message has its own variable-length record instead of a fixed
/// 2432-byte frame. Message 29 (model data) and every extended-size message
/// are variable length, like the walker treats them; message 31 framing is
/// decided by the caller (fixed inside the metadata record, variable after
/// it). Without this, a message 29 was skipped as one 2432-byte frame and
/// its remaining bytes were parsed as message headers and radials.
fn message_framing(header: &MessageHeader) -> (usize, bool) {
    let total_len = header.message_len();
    let variable = header.has_extended_size() || header.message_type == 29;
    (total_len, variable)
}

/// Parse the 16-byte message header at `offset`.
pub fn parse_message_header(bytes: &[u8], offset: usize) -> Result<MessageHeader> {
    require_len(bytes, offset, MESSAGE_HEADER_LEN, "message header")?;
    Ok(parse_message_header_bytes(
        &bytes[offset..offset + MESSAGE_HEADER_LEN],
    ))
}

fn parse_message_header_bytes(bytes: &[u8]) -> MessageHeader {
    debug_assert!(bytes.len() >= MESSAGE_HEADER_LEN);
    MessageHeader {
        size_halfwords: be_u16(bytes, 0),
        channels: bytes[2],
        message_type: bytes[3],
        sequence_id: be_u16(bytes, 4),
        date: be_u16(bytes, 6),
        milliseconds: be_u32(bytes, 8),
        segments: be_u16(bytes, 12),
        segment_number: be_u16(bytes, 14),
    }
}

fn parse_message_1(
    body: &[u8],
    message_header: &MessageHeader,
    builder: &mut VolumeBuilder,
) -> Result<()> {
    require_len(body, 0, MSG_1_HEADER_LEN, "message 1 header")?;

    let collect_ms = be_u32(body, 0);
    let collect_date = be_u16(body, 4);
    if builder.decoded_radials() == 0 && collect_date > 0 {
        builder.set_header_time(nexrad_date_ms_to_datetime(
            u32::from(collect_date),
            collect_ms,
        ));
    }

    let azimuth_angle = legacy_binary_angle_deg(be_u16(body, 8));
    let radial_status = RadialStatus::from(be_u16(body, 12) as u8);
    let elevation_angle = legacy_binary_angle_deg(be_u16(body, 14));
    let elevation_number = (be_u16(body, 16) as u8).max(1);

    let reflectivity_gates = BlockGates {
        first_gate_m: i32::from(be_i16(body, 18)),
        gate_spacing_m: i32::from(be_u16(body, 22).max(1)),
        gate_count: usize::from(be_u16(body, 26)),
    };
    let doppler_gates = BlockGates {
        first_gate_m: i32::from(be_i16(body, 20)),
        gate_spacing_m: i32::from(be_u16(body, 24).max(1)),
        gate_count: usize::from(be_u16(body, 28)),
    };
    let reflectivity_pointer = usize::from(be_u16(body, 36));
    let velocity_pointer = usize::from(be_u16(body, 38));
    let spectrum_width_pointer = usize::from(be_u16(body, 40));
    let velocity_resolution = be_u16(body, 42);

    builder.set_vcp(be_u16(body, 44));
    // Table III halfword 4: unambiguous range in 0.1 km; halfword 31: Nyquist
    // velocity in 0.01 m/s (bytes 6 and 60 of the body, as MetPy and Py-ART
    // read them). Halfwords 24 to 30 are spare.
    let unambiguous_range_m = match be_u16(body, 6) {
        0 => None,
        raw => Some(f32::from(raw) * 100.0),
    };
    let nyquist_velocity_mps = match be_i16(body, 60) {
        raw if raw > 0 => Some(raw as f32 / 100.0),
        _ => None,
    };

    let reflectivity_row =
        legacy_message_1_row(body, reflectivity_pointer, reflectivity_gates.gate_count);
    let velocity_row = legacy_message_1_row(body, velocity_pointer, doppler_gates.gate_count);
    let spectrum_width_row =
        legacy_message_1_row(body, spectrum_width_pointer, doppler_gates.gate_count);
    if reflectivity_row.is_none() && velocity_row.is_none() && spectrum_width_row.is_none() {
        builder.count_skipped();
        return Ok(());
    }
    for row in [reflectivity_row, velocity_row, spectrum_width_row]
        .into_iter()
        .flatten()
    {
        bounded_read::check_gate_count(row.len(), "message 1 moment")
            .map_err(NexradError::LimitExceeded)?;
    }

    // Table III values with no FM301 coordinate (offsets as MetPy and
    // Py-ART read them): radial status (halfword 7), azimuth number (6), cut
    // sector number (16), calibration constant (17-18, Real*4), atmospheric
    // attenuation (32), TOVER (33) and spot blanking status (34).
    let extras = RadialExtras {
        message: RadialMessage::Legacy,
        message_channels: message_header.channels,
        message_header: RadialMessageHeader::of(message_header),
        status_code: be_u16(body, 12),
        azimuth_number: be_u16(body, 10),
        cut_sector_number: be_u16(body, 30),
        spot_blanking: be_u16(body, 66),
        azimuth_spacing_code: None,
        azimuth_indexing_raw: None,
        radial: None,
        volume: None,
        elevation: None,
        zdr_bias_estimate_db: None,
        radar_identifier: None,
        legacy: Some(LegacyConstants {
            calibration_constant_db: be_f32(body, 32),
            atmospheric_attenuation_raw: be_i16(body, 62),
            tover_raw: be_i16(body, 64),
        }),
    };

    let sweep = builder.sweep_for_radial(radial_status, elevation_angle, elevation_number)?;
    let ray = builder.push_ray(
        sweep,
        collect_date,
        collect_ms,
        azimuth_angle,
        elevation_angle,
        nyquist_velocity_mps,
        unambiguous_range_m,
        radial_status,
        &extras,
        ONE_DEGREE_RADIALS_PER_CUT,
    )?;

    // ICD 2620002 message 1 encodes REF as (code - 66) / 2 and VEL and SW
    // like velocity at 0.5 m/s resolution ((code - 129) / 2), with code 0
    // below threshold and 1 range folded.
    let moments = [
        (reflectivity_row, *b"REF", reflectivity_gates, 2.0),
        (
            velocity_row,
            *b"VEL",
            doppler_gates,
            legacy_message_1_velocity_scale(velocity_resolution),
        ),
        (spectrum_width_row, *b"SW ", doppler_gates, 2.0),
    ];
    for (row, name, gates, scale) in moments {
        let Some(row) = row else {
            continue;
        };
        let block = MomentBlock {
            name,
            gates,
            scale,
            offset: if name == *b"REF" { 66.0 } else { 129.0 },
            row: MomentPayload::U8(row),
            // Message 1's legacy moment header has no TOVER, SNR threshold
            // or recombination code.
            extras: None,
        };
        builder.push_moment(sweep, ray, &block, ONE_DEGREE_RADIALS_PER_CUT)?;
    }

    builder.count_radial();
    Ok(())
}

fn legacy_message_1_row(body: &[u8], pointer: usize, gate_count: usize) -> Option<&[u8]> {
    if pointer == 0 || gate_count == 0 {
        return None;
    }
    body.get(pointer..pointer.checked_add(gate_count)?)
}

fn legacy_message_1_velocity_scale(velocity_resolution: u16) -> f32 {
    match velocity_resolution {
        2 => 2.0,
        4 => 1.0,
        _ => 2.0,
    }
}

fn legacy_binary_angle_deg(raw: u16) -> f32 {
    raw as f32 * 360.0 / 65_536.0
}

/// Decode one message 31 radial into `builder`: the sweep and ray indices
/// it went to.
fn parse_message_31(
    body: &[u8],
    message_header: &MessageHeader,
    builder: &mut VolumeBuilder,
) -> Result<(usize, usize)> {
    let header = parse_message_31_header(body, 0)?;
    let expected_radials = expected_radials_for_azimuth_resolution(header.azimuth_resolution);

    // GR2-style ".msg31" exports write a nonstandard volume-header date, so
    // the volume time parses as the epoch; recover it from the first
    // radial's collection time instead.
    if builder.header_time == DateTime::<Utc>::UNIX_EPOCH && header.collect_date > 0 {
        builder.set_header_time(nexrad_date_ms_to_datetime(
            u32::from(header.collect_date),
            header.collect_ms,
        ));
    }

    let mut nyquist_velocity_mps = None;
    let mut unambiguous_range_m = None;
    let mut moments: [Option<MomentBlock<'_>>; MAX_MESSAGE_31_MOMENTS] =
        std::array::from_fn(|_| None);
    let mut moment_count = 0;
    let needs_volume_constants = builder.needs_volume_constants();
    // Table XVII-A bytes 21 (the status with its bad-data bit), 28 and 29;
    // the header parse above checked the 72 bytes.
    let mut extras =
        RadialExtras::generic(body[21], header.azimuth_number, header.cut_sector, body[28]);
    extras.message_channels = message_header.channels;
    extras.message_header = RadialMessageHeader::of(message_header);
    extras.azimuth_spacing_code = Some(header.azimuth_resolution);
    extras.azimuth_indexing_raw = Some(body[29]);
    extras.radar_identifier = Some(header.radar_identifier);
    if builder.decoded_radials() == 0 {
        builder.record_radar_identifier(&header.radar_identifier);
    }

    for pointer in &header.block_pointers {
        if *pointer == 0 {
            continue;
        }
        let pointer = *pointer;
        if pointer > body.len().saturating_sub(4) {
            continue;
        }

        match body[pointer] {
            b'R' if &body[pointer + 1..pointer + 4] == b"VOL" => {
                extras.volume =
                    parse_volume_constant_block(body, pointer, builder, needs_volume_constants)?;
            }
            b'R' if &body[pointer + 1..pointer + 4] == b"ELV" => {
                extras.elevation = parse_elevation_constant_block(body, pointer);
            }
            b'R' if &body[pointer + 1..pointer + 4] == b"RAD" => {
                let (nyquist, unambiguous, constants) = parse_radial_constant_block(body, pointer)?;
                nyquist_velocity_mps = nyquist;
                unambiguous_range_m = unambiguous;
                extras.radial = Some(constants);
            }
            b'D' if moment_count < moments.len() => {
                moments[moment_count] = Some(parse_generic_moment_block(body, pointer)?);
                moment_count += 1;
            }
            _ => {}
        }
    }

    extras.zdr_bias_estimate_db = extras
        .volume
        .and_then(|volume| volume.zdr_bias_estimate_raw)
        .filter(|raw| *raw != 0)
        .map(|raw| {
            // Table XVII-E notes 20 and 33: encoded like this radial's ZDR
            // moment, with the Table XVII-I typical scale and offset when the
            // radial has none.
            let zdr = moments[..moment_count]
                .iter()
                .flatten()
                .find(|moment| moment.name == *b"ZDR" && moment.scale != 0.0);
            let (scale, offset) = zdr.map_or(
                (
                    messages::msg31_blocks::ZDR_SCALE,
                    messages::msg31_blocks::ZDR_OFFSET,
                ),
                |moment| (moment.scale, moment.offset),
            );
            (f32::from(raw) - offset) / scale
        });

    let sweep = builder.sweep_for_radial(
        header.radial_status,
        header.elevation_angle,
        header.elevation_number,
    )?;
    let ray = builder.push_ray(
        sweep,
        header.collect_date,
        header.collect_ms,
        header.azimuth_angle,
        header.elevation_angle,
        nyquist_velocity_mps,
        unambiguous_range_m,
        header.radial_status,
        &extras,
        expected_radials,
    )?;

    // Iterate by reference: moving the whole `[Option<MomentBlock>; 10]`
    // array into an iterator copied it for every radial.
    for moment in moments[..moment_count].iter().flatten() {
        builder.push_moment(sweep, ray, moment, expected_radials)?;
    }

    builder.count_radial();
    Ok((sweep, ray))
}

fn expected_radials_for_azimuth_resolution(azimuth_resolution: u8) -> usize {
    match azimuth_resolution {
        1 => HALF_DEGREE_RADIALS_PER_CUT,
        2 => ONE_DEGREE_RADIALS_PER_CUT,
        _ => FALLBACK_RADIALS_PER_CUT,
    }
}

impl Message31Header {
    /// The radar identifier: the message 31 identifier with spaces and NULs
    /// trimmed, or, when that is blank, `volume_header_icao` trimmed (which
    /// may be empty too).
    pub fn radar_identifier_or(&self, volume_header_icao: &str) -> String {
        radar_identifier_or(&self.radar_identifier, volume_header_icao)
    }
}

/// A message 31 radar identifier, falling back to the volume header ICAO when
/// the identifier is blank (spaces or NULs), and then to the empty string.
pub(crate) fn radar_identifier_or(identifier: &[u8; 4], volume_header_icao: &str) -> String {
    let identifier = ascii_trim(identifier);
    if identifier.is_empty() {
        volume_header_icao
            .trim_matches(|c: char| c == '\0' || c.is_whitespace())
            .to_owned()
    } else {
        identifier
    }
}

/// Parse the Data Header Block of a message 31 body. A blank radar
/// identifier (spaces or NULs) is accepted: real files have one (KVWX
/// 2008-04-15), and MetPy and Py-ART read them.
pub fn parse_message_31_header(bytes: &[u8], offset: usize) -> Result<Message31Header> {
    require_len(bytes, offset, MSG_31_HEADER_LEN, "message 31 header")?;
    let bytes = &bytes[offset..offset + MSG_31_HEADER_LEN];

    let mut block_pointers = [0; 10];
    for (index, pointer) in block_pointers.iter_mut().enumerate() {
        *pointer = be_u32(bytes, 32 + index * 4) as usize;
    }

    Ok(Message31Header {
        radar_identifier: [bytes[0], bytes[1], bytes[2], bytes[3]],
        collect_ms: be_u32(bytes, 4),
        collect_date: be_u16(bytes, 8),
        azimuth_number: be_u16(bytes, 10),
        azimuth_angle: be_f32(bytes, 12),
        radial_length: be_u16(bytes, 18),
        azimuth_resolution: bytes[20],
        radial_status: RadialStatus::from(bytes[21]),
        elevation_number: bytes[22],
        cut_sector: bytes[23],
        elevation_angle: be_f32(bytes, 24),
        block_pointers,
    })
}

/// The Volume Data Constant block (Table XVII-E). While the station location
/// or VCP is still unknown (`set_location`) it also sets them, and a block
/// too short for the 44-byte layout is an error, as it always was; on later
/// radials such a block is skipped.
fn parse_volume_constant_block(
    bytes: &[u8],
    offset: usize,
    builder: &mut VolumeBuilder,
    set_location: bool,
) -> Result<Option<VolumeConstants>> {
    if let Err(err) = require_len(
        bytes,
        offset,
        VOLUME_CONSTANT_BLOCK_LEN,
        "volume constant block",
    ) {
        return if set_location { Err(err) } else { Ok(None) };
    }
    let declared = usize::from(be_u16(bytes, offset + 4));
    let with_bias = declared >= VOLUME_CONSTANT_BLOCK_ZDR_BIAS_LEN
        && bytes.len() - offset >= VOLUME_CONSTANT_BLOCK_ZDR_BIAS_LEN;
    let block = &bytes[offset..offset + VOLUME_CONSTANT_BLOCK_LEN];
    if set_location {
        let location = &mut builder.volume.location;
        location.latitude_deg = Some(f64::from(be_f32(block, 8)));
        location.longitude_deg = Some(f64::from(be_f32(block, 12)));

        // Antenna height above MSL: tower plus feedhorn, as xradar and Py-ART.
        let tower_height_m = be_i16(block, 16) as f32;
        let feedhorn_height_m = be_u16(block, 18) as f32;
        location.altitude_m = Some(f64::from(tower_height_m + feedhorn_height_m));

        builder.set_vcp(be_u16(block, 40));
    }
    Ok(Some(VolumeConstants {
        version_major: block[6],
        version_minor: block[7],
        latitude_deg: be_f32(block, 8),
        longitude_deg: be_f32(block, 12),
        site_height_m: be_i16(block, 16),
        feedhorn_height_m: be_u16(block, 18),
        calibration_constant_db: be_f32(block, 20),
        horizontal_tx_power_kw: be_f32(block, 24),
        vertical_tx_power_kw: be_f32(block, 28),
        system_zdr_db: be_f32(block, 32),
        initial_system_phidp_deg: be_f32(block, 36),
        vcp_number: be_u16(block, 40),
        processing_status: be_u16(block, 42),
        zdr_bias_estimate_raw: with_bias.then(|| be_u16(bytes, offset + 44)),
    }))
}

/// The Elevation Data Constant block (Table XVII-F), or `None` when it runs
/// past the radial.
fn parse_elevation_constant_block(bytes: &[u8], offset: usize) -> Option<ElevationConstants> {
    let block = bytes.get(offset..offset.checked_add(ELEVATION_CONSTANT_BLOCK_LEN)?)?;
    Some(ElevationConstants {
        atmospheric_attenuation_raw: be_i16(block, 6),
        calibration_constant_db: be_f32(block, 8),
    })
}

/// The Radial Data Constant block (Table XVII-H): Nyquist velocity (0.01 m/s
/// at bytes 16-17) and unambiguous range (0.1 km at bytes 6-7), each `None`
/// when zero, and the noise levels, radial flags and (in the 28-byte layout)
/// calibration constants.
fn parse_radial_constant_block(
    bytes: &[u8],
    offset: usize,
) -> Result<(Option<f32>, Option<f32>, RadialBlockConstants)> {
    require_len(
        bytes,
        offset,
        RADIAL_CONSTANT_BLOCK_LEN,
        "radial constant block",
    )?;
    let block = &bytes[offset..offset + RADIAL_CONSTANT_BLOCK_LEN];
    let nyquist = be_i16(block, 16);
    let unambiguous = be_u16(block, 6);
    let declared = usize::from(be_u16(block, 4));
    let calibration_dbz = (declared >= RADIAL_CONSTANT_BLOCK_CALIBRATED_LEN
        && bytes.len() - offset >= RADIAL_CONSTANT_BLOCK_CALIBRATED_LEN)
        .then(|| (be_f32(bytes, offset + 20), be_f32(bytes, offset + 24)));
    Ok((
        (nyquist > 0).then_some(nyquist as f32 / 100.0),
        (unambiguous > 0).then_some(f32::from(unambiguous) * 100.0),
        RadialBlockConstants {
            horizontal_noise_level_dbm: be_f32(block, 8),
            vertical_noise_level_dbm: be_f32(block, 12),
            radial_flags: be_u16(block, 18),
            calibration_dbz,
        },
    ))
}

fn parse_generic_moment_block(bytes: &[u8], offset: usize) -> Result<MomentBlock<'_>> {
    require_len(
        bytes,
        offset,
        GENERIC_DATA_BLOCK_LEN,
        "generic moment block",
    )?;
    let header = &bytes[offset..offset + GENERIC_DATA_BLOCK_LEN];
    let name = [header[1], header[2], header[3]];
    let gate_count = usize::from(be_u16(header, 8));
    bounded_read::check_gate_count(gate_count, "message 31 moment block")
        .map_err(NexradError::LimitExceeded)?;
    let first_gate_m = i32::from(be_i16(header, 10));
    let gate_spacing_m = i32::from(be_i16(header, 12));
    let word_size = header[19];
    let scale = be_f32(header, 20);
    let offset_value = be_f32(header, 24);
    let data_offset = offset + GENERIC_DATA_BLOCK_LEN;

    let row = match word_size {
        8 => {
            require_len(bytes, data_offset, gate_count, "8-bit moment gates")?;
            MomentPayload::U8(&bytes[data_offset..data_offset + gate_count])
        }
        16 => {
            // `ok_or_else`: `ok_or` built this error's `String` for every
            // 16-bit block of every radial.
            let byte_count =
                gate_count
                    .checked_mul(2)
                    .ok_or_else(|| NexradError::InvalidMessage {
                        offset,
                        reason: "16-bit moment gate count overflow".to_owned(),
                    })?;
            require_len(bytes, data_offset, byte_count, "16-bit moment gates")?;
            MomentPayload::U16(&bytes[data_offset..data_offset + byte_count])
        }
        other => {
            return Err(NexradError::InvalidMessage {
                offset,
                reason: format!("unsupported moment word size {other}"),
            });
        }
    };

    Ok(MomentBlock {
        name,
        gates: BlockGates {
            first_gate_m,
            gate_spacing_m,
            gate_count,
        },
        scale,
        offset: offset_value,
        row,
        extras: Some(MomentHeaderExtras {
            tover_raw: be_u16(header, 14),
            snr_threshold_raw: be_i16(header, 16),
            control_flags: header[18],
        }),
    })
}

fn nexrad_date_ms_to_datetime(date: u32, milliseconds: u32) -> DateTime<Utc> {
    let days = i64::from(date.saturating_sub(1));
    let seconds = days * 86_400 + i64::from(milliseconds / 1000);
    let nanos = (milliseconds % 1000) * 1_000_000;
    Utc.timestamp_opt(seconds, nanos)
        .single()
        .unwrap_or(DateTime::<Utc>::UNIX_EPOCH)
}

fn ascii_trim(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim_matches(char::from(0))
        .trim()
        .to_owned()
}

fn require_len(bytes: &[u8], offset: usize, needed: usize, what: &'static str) -> Result<()> {
    let available = bytes.len().saturating_sub(offset);
    if available < needed {
        Err(NexradError::Truncated {
            what,
            offset,
            needed,
            available,
        })
    } else {
        Ok(())
    }
}

fn be_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes([bytes[offset], bytes[offset + 1]])
}

fn be_i16(bytes: &[u8], offset: usize) -> i16 {
    i16::from_be_bytes([bytes[offset], bytes[offset + 1]])
}

fn be_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn be_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(be_u32(bytes, offset))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    require_len(bytes, offset, 4, "u32")?;
    Ok(u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

fn i32_at(bytes: &[u8], offset: usize) -> Result<i32> {
    require_len(bytes, offset, 4, "i32")?;
    Ok(i32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

#[cfg(test)]
mod tests {
    //! Decoder tests on real Level II files from the corpus
    //! (`recast-radar-testdata`, ids from `testdata/level2/manifest.toml`).
    //!
    //! Expected values come from `testdata/level2/golden/decode/<name>.json`,
    //! written by `tools/level2_decode_golden.py`: a byte walker written for the
    //! script (volume header, LDM record framing, message headers, Message 1
    //! and Message 31 radial headers and raw gate codes), MetPy 1.7.1
    //! `Level2File` (station, times, sweeps, radial headers, VOL block, Nyquist,
    //! VCP, scaled moments) and Py-ART 2.2.5 `NEXRADLevel2File` (rays per
    //! sweep, raw codes, Nyquist). The script fails unless the readers agree.
    //!
    //! Edge and corruption cases change bytes of the real files; each test
    //! shows the change. Inputs that are not committed are downloaded on first
    //! use, and their tests are skipped when that is not possible.

    use super::*;
    use bzip2::read::BzDecoder;
    use recast_radar_core::model::{
        Field, FieldData, FieldName, LinearTransform, RangeCoord, Sweep,
    };
    use serde_json::Value;
    use std::collections::BTreeSet;
    use std::io::Write;

    /// Build 22.0 (VOL 52, RAD 28, ZDR and PHI 16-bit, CFP), LDM bzip2
    /// records: the metadata record and 480 + 480 radials of the split cut.
    const KTLX_2024_TRIM: &str = "l2-ktlx-20240315-000217-trim";
    /// The full volume the trim was cut from (downloaded).
    const KTLX_2024_FULL: &str = "l2-ktlx-20240315-000217";
    /// Build 13.2 (68-byte Message 31 header, PHI 16-bit, ZDR 8-bit).
    const KTLX_2013_TRIM: &str = "l2-ktlx-20130520-201643-trim";
    /// ARCHIVE2.001 with a blank ICAO, Message 1 radials.
    const KTLX_1991_TRIM: &str = "l2-ktlx-19910605-162126-trim";
    /// ARCHIVE2.036 with a NUL ICAO, Message 1 radials, VCP 11.
    const KTLX_1999_TRIM: &str = "l2-ktlx-19990504-002218-trim";
    /// gzip archive object, Message 31 at 1 degree, 7 sweeps (downloaded).
    const KPAH_2008_GZIP: &str = "l2-kpah-20080415-235014";
    /// gzip archive object that ends 68 radials into its first cut (downloaded).
    const KTLX_1999_TRUNCATED_GZIP: &str = "l2-ktlx-19990503-230052";
    /// Committed real-time chunks of KIWA volume 307: the Start chunk and the
    /// first intermediate chunk (the first 120 radials of a 720-radial cut).
    const KIWA_CHUNK_START: &str = "l2chunk-kiwa-307-20260917-003629-001-s";
    const KIWA_CHUNK_002: &str = "l2chunk-kiwa-307-20260917-003629-002-i";
    const KIWA_CHUNK_003: &str = "l2chunk-kiwa-307-20260917-003629-003-i";
    const KIWA_CHUNKS_GOLDEN: &str = "l2chunk-kiwa-307-20260917-003629-001-s+002-i";

    // ------------------------------------------------------------ inputs ---

    /// Bytes of a corpus file, or `None` (with a message) when it is not
    /// committed and cannot be downloaded right now.
    fn corpus_bytes(id: &str) -> Option<Vec<u8>> {
        match recast_radar_testdata::bytes(id) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.is_offline() => {
                eprintln!("skipping: {error}");
                None
            }
            Err(error) => panic!("{error}"),
        }
    }

    /// Golden values for a corpus input (see the module docs).
    fn golden(name: &str) -> Value {
        let path = recast_radar_testdata::testdata_dir()
            .join("level2")
            .join("golden")
            .join("decode")
            .join(format!("{name}.json"));
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        serde_json::from_str(&text)
            .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
    }

    fn int(value: &Value) -> i64 {
        value
            .as_i64()
            .unwrap_or_else(|| panic!("expected an integer, found {value}"))
    }

    fn count(value: &Value) -> usize {
        usize::try_from(int(value)).unwrap_or_else(|_| panic!("expected a count, found {value}"))
    }

    fn num(value: &Value) -> f64 {
        value
            .as_f64()
            .unwrap_or_else(|| panic!("expected a number, found {value}"))
    }

    fn text(value: &Value) -> &str {
        value
            .as_str()
            .unwrap_or_else(|| panic!("expected a string, found {value}"))
    }

    fn list(value: &Value) -> &[Value] {
        value
            .as_array()
            .unwrap_or_else(|| panic!("expected an array, found {value}"))
    }

    /// Header text as the decoder reports it: NUL padding and blanks trimmed.
    fn trimmed(text: &str) -> &str {
        text.trim_matches(|c: char| c == '\0' || c.is_whitespace())
    }

    fn hex_text(hex: &str) -> String {
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte"))
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64, what: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what}: {actual} vs {expected} (tolerance {tolerance})"
        );
    }

    /// `(control word offset, compressed payload)` of each LDM record after
    /// the volume header, read from the file's control words.
    fn ldm_records(file: &[u8]) -> Vec<(usize, &[u8])> {
        let mut records = Vec::new();
        let mut offset = VOLUME_HEADER_LEN;
        while offset + 4 <= file.len() {
            let control = i32::from_be_bytes(file[offset..offset + 4].try_into().unwrap());
            let len = control.unsigned_abs() as usize;
            records.push((offset, &file[offset + 4..offset + 4 + len]));
            offset += 4 + len;
            if control < 0 {
                break;
            }
        }
        records
    }

    fn bzip2_decompress(block: &[u8]) -> Vec<u8> {
        let mut decoded = Vec::new();
        BzDecoder::new(block)
            .read_to_end(&mut decoded)
            .expect("real LDM record decompresses");
        decoded
    }

    fn bzip2_compress(payload: &[u8]) -> Vec<u8> {
        let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::best());
        encoder.write_all(payload).unwrap();
        encoder.finish().unwrap()
    }

    /// Stream offsets of the messages in an uncompressed Message 31 run
    /// (control word + message, back to back), plus the end offset.
    fn message31_bounds(stream: &[u8]) -> Vec<usize> {
        let mut bounds = vec![0];
        let mut offset = 0;
        while offset < stream.len() {
            let header = parse_message_header(stream, offset + CONTROL_WORD_LEN).unwrap();
            assert_eq!(header.message_type, 31, "message at {offset}");
            offset += CONTROL_WORD_LEN + usize::from(header.size_halfwords) * 2;
            bounds.push(offset);
        }
        assert_eq!(offset, stream.len());
        bounds
    }

    // ------------------------------------------------------ golden checks ---

    /// A decoded volume with the decode state the FM301 model does not keep:
    /// the Archive II volume header time (replaced by the first Message 1
    /// radial's time, as the golden script reads it) and each sweep's first
    /// and last radial status.
    struct Decoded {
        volume: Volume,
        header_time: DateTime<Utc>,
        statuses: Vec<(Option<RadialStatus>, Option<RadialStatus>)>,
    }

    fn finished(builder: VolumeBuilder) -> Decoded {
        let header_time = builder.header_time;
        let (volume, states) = builder.finish().unwrap();
        let statuses = states
            .iter()
            .map(|state| (state.first_status, state.last_status))
            .collect();
        Decoded {
            volume,
            header_time,
            statuses,
        }
    }

    fn decode(bytes: &[u8]) -> Decoded {
        finished(builder_from_bytes(bytes).unwrap())
    }

    /// Structural equality with NaN equal to NaN: per-ray Nyquist and
    /// unambiguous range hold NaN for rays without a value, so values are
    /// compared with those NaNs replaced.
    trait Comparable: Clone + PartialEq {
        fn without_nan(&self) -> Self;
    }

    impl Comparable for Sweep {
        fn without_nan(&self) -> Self {
            let mut sweep = self.clone();
            for values in [
                &mut sweep.ray_vars.nyquist_velocity_mps,
                &mut sweep.ray_vars.unambiguous_range_m,
            ]
            .into_iter()
            .flatten()
            {
                for value in values.iter_mut().filter(|value| value.is_nan()) {
                    *value = f32::MAX;
                }
            }
            if let Some(monitoring) = sweep.monitoring.as_deref_mut() {
                for values in [
                    &mut monitoring.radar_measured_transmit_power_h_dbm,
                    &mut monitoring.radar_measured_transmit_power_v_dbm,
                ]
                .into_iter()
                .flatten()
                {
                    for value in values.iter_mut().filter(|value| value.is_nan()) {
                        *value = f32::MAX;
                    }
                }
            }
            sweep
        }
    }

    impl Comparable for Vec<Sweep> {
        fn without_nan(&self) -> Self {
            self.iter().map(Sweep::without_nan).collect()
        }
    }

    impl Comparable for Volume {
        fn without_nan(&self) -> Self {
            let mut volume = self.clone();
            volume.sweeps = volume.sweeps.without_nan();
            volume
        }
    }

    fn assert_same<T: Comparable>(left: &T, right: &T, what: &str) {
        assert!(
            left.without_nan() == right.without_nan(),
            "{what}: values differ"
        );
    }

    /// The ICD moment name of a field (the golden's keys).
    fn icd_name(name: &FieldName) -> &str {
        match name {
            FieldName::Dbzh => "REF",
            FieldName::Vradh => "VEL",
            FieldName::Wradh => "SW",
            FieldName::Zdr => "ZDR",
            FieldName::Phidp => "PHI",
            FieldName::Rhohv => "RHO",
            FieldName::Ccorh => "CFP",
            other => other.as_str(),
        }
    }

    fn field<'v>(volume: &'v Volume, sweep: usize, name: &FieldName) -> &'v Field {
        volume.sweeps[sweep]
            .field(name)
            .unwrap_or_else(|| panic!("sweep {sweep} has no {name}"))
    }

    fn code_at(field: &Field, ray: usize, gate: usize) -> u16 {
        let index = ray * field.ngates as usize + gate;
        match &field.data {
            FieldData::U8 { values, .. } => u16::from(values[index]),
            FieldData::U16 { values, .. } => values[index],
            _ => panic!("Level II moments are stored as integer codes"),
        }
    }

    fn word_size_bits(field: &Field) -> i64 {
        match &field.data {
            FieldData::U8 { .. } => 8,
            FieldData::U16 { .. } => 16,
            _ => panic!("Level II moments are stored as integer codes"),
        }
    }

    fn icd_scale_offset(field: &Field) -> (f32, f32) {
        let transform = match &field.data {
            FieldData::U8 { coding, .. } => coding.transform,
            FieldData::U16 { coding, .. } => coding.transform,
            _ => panic!("Level II moments are stored as integer codes"),
        };
        match transform {
            LinearTransform::IcdScaleOffset { scale, offset } => (scale, offset),
            other => panic!("Level II moments use the ICD transform, not {other:?}"),
        }
    }

    /// One moment against its golden summary: the rays that carry it, word
    /// size, gate layout, scaling, the count and sum of valid raw codes, the
    /// scaled sum, minimum and maximum, and sampled gates (raw code and
    /// scaled value). Golden rows index the rays that carry the moment.
    fn assert_moment_matches_golden(
        field: &Field,
        range: &RangeCoord,
        golden: &Value,
        label: &str,
    ) {
        let rows = count(&golden["rows"]);
        let present: Vec<usize> = (0..field.nrays as usize)
            .filter(|ray| !field.is_absent(*ray))
            .collect();
        assert_eq!(present.len(), rows, "{label}: rows");
        let radial_indices: Vec<usize> = match golden["row_radials"].as_array() {
            Some(indices) => indices.iter().map(count).collect(),
            None => (0..rows).collect(),
        };
        assert_eq!(present, radial_indices, "{label}: rays with the moment");
        assert_eq!(
            word_size_bits(field),
            int(&golden["word_size"]),
            "{label}: word size"
        );
        let (first_gate_m, spacing_m) = field.native_geometry(range).expect("gate geometry");
        assert_eq!(
            first_gate_m,
            num(&golden["first_gate_m"]),
            "{label}: first gate centre"
        );
        assert_eq!(
            spacing_m,
            num(&golden["gate_width_m"]),
            "{label}: gate spacing"
        );
        assert_eq!(
            field.ngates as usize,
            count(&golden["gates_max"]),
            "{label}: gates"
        );
        let (scale, offset) = icd_scale_offset(field);
        assert_eq!(scale, num(&golden["scale"]) as f32, "{label}: scale");
        assert_eq!(offset, num(&golden["offset"]) as f32, "{label}: offset");

        let mut valid = 0usize;
        let mut raw_sum = 0u64;
        let mut scaled_sum = 0.0f64;
        let mut abs_sum = 0.0f64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &ray in &present {
            for gate in 0..field.ngates as usize {
                let code = code_at(field, ray, gate);
                match field.value(ray, gate) {
                    Some(value) => {
                        assert!(
                            code >= 2,
                            "{label}: code {code} at {ray}/{gate} has a value"
                        );
                        let value = f64::from(value);
                        valid += 1;
                        raw_sum += u64::from(code);
                        scaled_sum += value;
                        abs_sum += value.abs();
                        min = min.min(value);
                        max = max.max(value);
                    }
                    None => assert!(code < 2, "{label}: code {code} at {ray}/{gate} is missing"),
                }
            }
        }
        assert_eq!(valid, count(&golden["valid_count"]), "{label}: valid gates");
        assert_eq!(
            raw_sum,
            u64::try_from(int(&golden["raw_valid_sum"])).unwrap(),
            "{label}: raw code sum"
        );
        // Scaled values are f32 here and float64 in MetPy: allow the f32
        // rounding of each value.
        assert_close(
            scaled_sum,
            num(&golden["scaled_sum"]),
            2e-7 * abs_sum + 1e-9,
            &format!("{label}: scaled sum"),
        );
        if valid > 0 {
            assert_close(
                min,
                num(&golden["scaled_min"]),
                1e-4,
                &format!("{label}: min"),
            );
            assert_close(
                max,
                num(&golden["scaled_max"]),
                1e-4,
                &format!("{label}: max"),
            );
        }
        for sample in list(&golden["samples"]) {
            let sample = list(sample);
            let (row, gate) = (count(&sample[0]), count(&sample[1]));
            let ray = present[row];
            assert_eq!(
                i64::from(code_at(field, ray, gate)),
                int(&sample[2]),
                "{label}: code at {row}/{gate}"
            );
            match sample[3].as_f64() {
                Some(expected) => assert_close(
                    f64::from(field.value(ray, gate).expect("valid gate")),
                    expected,
                    1e-4,
                    &format!("{label}: value at {row}/{gate}"),
                ),
                None => assert_eq!(field.value(ray, gate), None, "{label}: {row}/{gate}"),
            }
        }
    }

    /// One sweep against a golden sweep: ray count, elevation number, first
    /// and last radial status (`statuses`, from the decoder state), angles,
    /// Nyquist velocities and every moment.
    fn assert_sweep_matches_golden(
        sweep: &Sweep,
        statuses: (Option<RadialStatus>, Option<RadialStatus>),
        golden: &Value,
        label: &str,
    ) {
        let radials = count(&golden["radials"]);
        assert_eq!(sweep.nrays(), radials, "{label}: radials");
        assert_eq!(
            sweep.elevation_number.map(i64::from),
            Some(int(&golden["elevation_number"])),
            "{label}: elevation number"
        );
        let status = |value: &Value| Some(RadialStatus::from(u8::try_from(int(value)).unwrap()));
        assert_eq!(
            statuses.0,
            status(&golden["first_status"]),
            "{label}: first status"
        );
        assert_eq!(
            statuses.1,
            status(&golden["last_status"]),
            "{label}: last status"
        );
        let rays = &sweep.rays;
        assert_close(
            f64::from(rays.elevation_deg[0]),
            num(&golden["first_elevation_deg"]),
            1e-4,
            &format!("{label}: elevation"),
        );
        assert_close(
            f64::from(rays.azimuth_deg[0]),
            num(&golden["first_azimuth_deg"]),
            1e-4,
            &format!("{label}: first azimuth"),
        );
        assert_close(
            f64::from(rays.azimuth_deg[radials - 1]),
            num(&golden["last_azimuth_deg"]),
            1e-4,
            &format!("{label}: last azimuth"),
        );
        let tolerance = 1e-4 * radials as f64;
        let azimuth_sum: f64 = rays.azimuth_deg.iter().map(|a| f64::from(*a)).sum();
        assert_close(
            azimuth_sum,
            num(&golden["azimuth_sum_deg"]),
            tolerance,
            &format!("{label}: azimuth sum"),
        );
        let elevation_sum: f64 = rays.elevation_deg.iter().map(|e| f64::from(*e)).sum();
        assert_close(
            elevation_sum,
            num(&golden["elevation_sum_deg"]),
            tolerance,
            &format!("{label}: elevation sum"),
        );

        let nyquist: Vec<f64> = sweep
            .ray_vars
            .nyquist_velocity_mps
            .iter()
            .flatten()
            .filter(|value| value.is_finite())
            .map(|value| f64::from(*value))
            .collect();
        assert_eq!(
            nyquist.len(),
            count(&golden["nyquist_count"]),
            "{label}: radials with a Nyquist velocity"
        );
        assert_close(
            nyquist.iter().sum(),
            num(&golden["nyquist_sum_mps"]),
            tolerance,
            &format!("{label}: Nyquist sum"),
        );
        if !nyquist.is_empty() {
            let min = nyquist.iter().copied().fold(f64::INFINITY, f64::min);
            let max = nyquist.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert_close(
                min,
                num(&golden["nyquist_min_mps"]),
                1e-4,
                &format!("{label}: Nyquist min"),
            );
            assert_close(
                max,
                num(&golden["nyquist_max_mps"]),
                1e-4,
                &format!("{label}: Nyquist max"),
            );
        }

        let moments = golden["moments"]
            .as_object()
            .unwrap_or_else(|| panic!("{label}: moments"));
        let names: BTreeSet<&str> = sweep.fields.iter().map(|f| icd_name(&f.name)).collect();
        let expected: BTreeSet<&str> = moments.keys().map(String::as_str).collect();
        assert_eq!(names, expected, "{label}: moments");
        for field in &sweep.fields {
            let name = icd_name(&field.name);
            assert_moment_matches_golden(
                field,
                &sweep.range,
                &moments[name],
                &format!("{label} {name}"),
            );
        }
    }

    /// A decoded volume against its golden file: station id, archive
    /// version, volume time and time reference, VCP, site, radial count and
    /// every sweep.
    fn assert_volume_matches_golden(decoded: &Decoded, golden: &Value) {
        let volume = &decoded.volume;
        let name = text(&golden["name"]);
        let header = &golden["volume_header"];
        assert_eq!(
            volume.attrs.instrument_name,
            trimmed(&hex_text(text(&header["icao_hex"]))),
            "{name}: station id"
        );
        let version = format!(
            "{}{}",
            trimmed(text(&header["tape"])),
            trimmed(text(&header["extension"]))
        );
        assert_eq!(
            volume.provenance.source_version.as_deref(),
            Some(version.as_str()),
            "{name}: archive version"
        );
        let sweeps = list(&golden["sweeps"]);
        // The header time; Message 1 volumes take it from the first radial.
        let expected_time = if int(&golden["first_radial"]["message_type"]) == 1 {
            int(&sweeps[0]["first_epoch_ms"])
        } else {
            int(&header["metpy_epoch_ms"])
        };
        assert_eq!(
            decoded.header_time.timestamp_millis(),
            expected_time,
            "{name}: volume time"
        );
        // The time reference is the first radial's time floored to the
        // second; ray times are offsets from it (design note 5.1).
        let first_ms = int(&sweeps[0]["first_epoch_ms"]);
        assert_eq!(
            volume.time_reference.timestamp_millis(),
            first_ms - first_ms.rem_euclid(1000),
            "{name}: time reference"
        );
        assert_eq!(
            volume.ray_time(0, 0).map(|time| time.timestamp_millis()),
            Some(first_ms),
            "{name}: first ray time"
        );
        // The decoder keeps the Message 1 VCP, else the VOL block VCP, else
        // the Message 5 pattern, ignoring zero.
        let expected_vcp = [
            &golden["message1_vcp"],
            &golden["site"]["vol_block_vcp"],
            &golden["message5_vcp"],
        ]
        .into_iter()
        .filter_map(Value::as_i64)
        .find(|pattern| *pattern != 0);
        assert_eq!(
            volume.scan.vcp_pattern.map(i64::from),
            expected_vcp,
            "{name}: VCP"
        );
        let location = &volume.location;
        match golden["site"].as_object() {
            Some(site) => {
                assert_eq!(
                    location.latitude_deg,
                    Some(f64::from(num(&site["latitude_deg"]) as f32)),
                    "{name}: latitude"
                );
                assert_eq!(
                    location.longitude_deg,
                    Some(f64::from(num(&site["longitude_deg"]) as f32)),
                    "{name}: longitude"
                );
                assert_eq!(
                    location.altitude_m,
                    Some((int(&site["site_amsl_m"]) + int(&site["feedhorn_agl_m"])) as f64),
                    "{name}: site height"
                );
            }
            None => {
                assert_eq!(location.latitude_deg, None, "{name}: latitude");
                assert_eq!(location.longitude_deg, None, "{name}: longitude");
                assert_eq!(location.altitude_m, None, "{name}: site height");
            }
        }
        let radials: usize = sweeps.iter().map(|sweep| count(&sweep["radials"])).sum();
        assert_eq!(
            volume.provenance.decode.decoded_ray_count, radials,
            "{name}: decoded radials"
        );
        assert_eq!(volume.sweeps.len(), sweeps.len(), "{name}: sweeps");
        for (index, (sweep, golden_sweep)) in volume.sweeps.iter().zip(sweeps).enumerate() {
            assert_sweep_matches_golden(
                sweep,
                decoded.statuses[index],
                golden_sweep,
                &format!("{name} sweep {index}"),
            );
        }
    }

    // ------------------------------------------------ headers and messages ---

    #[test]
    fn parses_archive_volume_header() {
        for id in [KTLX_2024_TRIM, KTLX_1991_TRIM, KTLX_1999_TRIM] {
            let Some(bytes) = corpus_bytes(id) else {
                return;
            };
            let expected = &golden(id)["volume_header"];

            let header = parse_volume_header(&bytes).unwrap();

            assert_eq!(
                header.archive_version,
                format!(
                    "{}{}",
                    trimmed(text(&expected["tape"])),
                    trimmed(text(&expected["extension"]))
                ),
                "{id}"
            );
            // KTLX; blank (1991) and NUL (1999) ICAO fields decode as empty.
            assert_eq!(header.icao, trimmed(text(&expected["metpy_stid"])), "{id}");
            assert_eq!(
                header.volume_time.timestamp_millis(),
                int(&expected["metpy_epoch_ms"]),
                "{id}"
            );
        }
        assert!(parse_volume_header(&[0; VOLUME_HEADER_LEN - 1]).is_err());
    }

    #[test]
    fn parses_message_header() {
        for id in [KTLX_2024_TRIM, KTLX_2013_TRIM, KTLX_1991_TRIM] {
            let Some(bytes) = corpus_bytes(id) else {
                return;
            };
            let golden = golden(id);
            let (normalized, compression) = normalize_archive_bytes(&bytes).unwrap();
            assert_eq!(compression, ArchiveCompression::Bzip2Blocks, "{id}");
            assert_eq!(
                normalized.len(),
                VOLUME_HEADER_LEN + count(&golden["stream_len"]),
                "{id}: decompressed length"
            );

            // Every metadata message header up to and including the first radial.
            let expected = list(&golden["messages_before_first_radial"]);
            assert!(!expected.is_empty());
            for message in expected {
                let fields: Vec<i64> = list(message).iter().map(int).collect();
                let offset = VOLUME_HEADER_LEN + usize::try_from(fields[0]).unwrap();
                let header = parse_message_header(&normalized, offset + CONTROL_WORD_LEN).unwrap();
                assert_eq!(
                    header,
                    MessageHeader {
                        size_halfwords: u16::try_from(fields[1]).unwrap(),
                        channels: u8::try_from(fields[2]).unwrap(),
                        message_type: u8::try_from(fields[3]).unwrap(),
                        sequence_id: u16::try_from(fields[4]).unwrap(),
                        date: u16::try_from(fields[5]).unwrap(),
                        milliseconds: u32::try_from(fields[6]).unwrap(),
                        segments: u16::try_from(fields[7]).unwrap(),
                        segment_number: u16::try_from(fields[8]).unwrap(),
                    },
                    "{id}: message at stream offset {}",
                    fields[0]
                );
            }

            // A header cut off by the end of the real data is an error.
            let end = normalized.len();
            assert!(matches!(
                parse_message_header(&normalized, end - MESSAGE_HEADER_LEN + 1),
                Err(NexradError::Truncated { .. })
            ));
        }
    }

    #[test]
    fn parses_message_31_header() {
        // Build 22.0 (72-byte header, ten pointers) and Build 13.2 (68-byte
        // header: the tenth "pointer" is the first word of the VOL block).
        for id in [KTLX_2024_TRIM, KTLX_2013_TRIM] {
            let Some(bytes) = corpus_bytes(id) else {
                return;
            };
            let expected = &golden(id)["first_radial"];
            assert_eq!(int(&expected["message_type"]), 31);
            let (normalized, _) = normalize_archive_bytes(&bytes).unwrap();
            let offset = VOLUME_HEADER_LEN + count(&expected["body_offset"]);

            let header = parse_message_31_header(&normalized, offset).unwrap();

            assert_eq!(
                i64::from(header.collect_ms),
                int(&expected["time_ms"]),
                "{id}"
            );
            assert_eq!(
                i64::from(header.collect_date),
                int(&expected["date"]),
                "{id}"
            );
            assert_eq!(
                i64::from(header.azimuth_number),
                int(&expected["azimuth_number"]),
                "{id}"
            );
            assert_eq!(
                header.azimuth_angle,
                num(&expected["azimuth_deg"]) as f32,
                "{id}"
            );
            assert_eq!(
                i64::from(header.radial_length),
                int(&expected["radial_length"]),
                "{id}"
            );
            assert_eq!(
                i64::from(header.azimuth_resolution),
                int(&expected["azimuth_spacing_code"]),
                "{id}"
            );
            assert_eq!(
                header.radial_status,
                RadialStatus::from(u8::try_from(int(&expected["status"])).unwrap()),
                "{id}"
            );
            assert_eq!(header.radial_status, RadialStatus::StartVolume, "{id}");
            assert_eq!(
                i64::from(header.elevation_number),
                int(&expected["elevation_number"]),
                "{id}"
            );
            assert_eq!(
                i64::from(header.cut_sector),
                int(&expected["cut_sector"]),
                "{id}"
            );
            assert_eq!(
                header.elevation_angle,
                num(&expected["elevation_deg"]) as f32,
                "{id}"
            );
            let pointers: Vec<usize> = list(&expected["block_pointers"])
                .iter()
                .map(count)
                .collect();
            assert_eq!(header.block_pointers.to_vec(), pointers, "{id}");
        }
    }

    // --------------------------------------------------- volume decoding ---

    #[test]
    fn decodes_message_31_volume() {
        let Some(bytes) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let golden = golden(KTLX_2024_TRIM);

        let decoded = decode(&bytes);

        let volume = &decoded.volume;
        assert_eq!(volume.attrs.instrument_name, "KTLX");
        assert_eq!(volume.scan.vcp_pattern, Some(212));
        assert_eq!(volume.scan.id, Some(212));
        assert_eq!(volume.scan.name.as_deref(), Some("VCP-212"));
        assert_eq!(volume.sweeps.len(), 2);
        assert_eq!(volume.sweeps[0].nrays(), 480);
        assert_eq!(
            volume.provenance.decode.message_count,
            count(&golden["message_count"])
        );
        assert_same(
            &read_volume_from_bytes(&bytes).unwrap(),
            volume,
            "public decode",
        );
        assert_volume_matches_golden(&decoded, &golden);
    }

    #[test]
    fn decodes_legacy_message_1_reflectivity_and_velocity() {
        let Some(bytes) = corpus_bytes(KTLX_1999_TRIM) else {
            return;
        };
        let golden = golden(KTLX_1999_TRIM);

        let decoded = decode(&bytes);

        // REF-only surveillance cut, then the Doppler cut (VEL/SW at 250 m)
        // whose radials carry the Nyquist velocity (Message 1 halfword 31).
        let volume = &decoded.volume;
        assert_eq!(volume.scan.vcp_pattern, Some(11));
        let surveillance = &volume.sweeps[0];
        let doppler = &volume.sweeps[1];
        assert!(surveillance.field(&FieldName::Dbzh).is_some());
        assert!(surveillance.field(&FieldName::Vradh).is_none());
        assert!(doppler.field(&FieldName::Vradh).is_some());
        assert_eq!(surveillance.ray_vars.nyquist_velocity_mps, None);
        let nyquist = doppler
            .ray_vars
            .nyquist_velocity_mps
            .as_deref()
            .expect("Doppler radials carry a Nyquist velocity");
        assert_eq!(
            nyquist[0],
            num(&golden["sweeps"][1]["nyquist_min_mps"]) as f32
        );
        // Both cuts share the elevation: REF 1 km gates centred from 0 m on
        // the surveillance cut, VEL/SW 250 m gates from -375 m on the other.
        assert_eq!(surveillance.range.spacing_m(), Some(1000.0));
        assert_eq!(surveillance.range.center_m(0), Some(0.0));
        assert_eq!(doppler.range.spacing_m(), Some(250.0));
        assert_eq!(doppler.range.center_m(0), Some(-375.0));
        assert_volume_matches_golden(&decoded, &golden);
    }

    #[test]
    fn decodes_legacy_message_1_spectrum_width_with_velocity_offset() {
        let Some(bytes) = corpus_bytes(KTLX_1999_TRIM) else {
            return;
        };
        let expected = &golden(KTLX_1999_TRIM)["sweeps"][1]["moments"]["SW"];

        let decoded = decode(&bytes);

        // ICD 2620002: SW = (code - 129) / 2, as MetPy and Py-ART decode it.
        let sweep = &decoded.volume.sweeps[1];
        let spectrum_width = field(&decoded.volume, 1, &FieldName::Wradh);
        assert_eq!(icd_scale_offset(spectrum_width), (2.0, 129.0));
        for sample in list(&expected["samples"]) {
            let sample = list(sample);
            let (row, gate, code) = (count(&sample[0]), count(&sample[1]), int(&sample[2]));
            if code >= 2 {
                assert_eq!(
                    spectrum_width.value(row, gate),
                    Some((code as f32 - 129.0) / 2.0)
                );
            }
        }
        assert_moment_matches_golden(spectrum_width, &sweep.range, expected, "KTLX 1999 SW");
    }

    #[test]
    fn decodes_16_bit_moments() {
        // KTLX 2013: PHI 16-bit, ZDR 8-bit. KTLX 2024: PHI and ZDR 16-bit.
        for (id, zdr_bits) in [(KTLX_2013_TRIM, 8), (KTLX_2024_TRIM, 16)] {
            let Some(bytes) = corpus_bytes(id) else {
                return;
            };
            let golden = golden(id);
            let decoded = decode(&bytes);
            let sweep = &decoded.volume.sweeps[0];
            let moments = &golden["sweeps"][0]["moments"];

            let phi = field(&decoded.volume, 0, &FieldName::Phidp);
            let zdr = field(&decoded.volume, 0, &FieldName::Zdr);
            assert_eq!(word_size_bits(phi), 16, "{id}");
            assert_eq!(word_size_bits(zdr), zdr_bits, "{id}");
            assert_moment_matches_golden(phi, &sweep.range, &moments["PHI"], &format!("{id} PHI"));
            assert_moment_matches_golden(zdr, &sweep.range, &moments["ZDR"], &format!("{id} ZDR"));
        }
    }

    #[test]
    fn decodes_every_trimmed_fixture() {
        let ids = recast_radar_testdata::ids_with_tag("trimmed");
        assert!(ids.len() >= 16, "trimmed fixtures: {ids:?}");
        for id in ids {
            let Some(bytes) = corpus_bytes(id) else {
                return;
            };
            let decoded = finished(
                builder_from_bytes(&bytes).unwrap_or_else(|error| panic!("{id}: {error}")),
            );
            assert_eq!(
                decoded.volume.provenance.compression.as_deref(),
                Some("bzip2-blocks")
            );
            assert_volume_matches_golden(&decoded, &golden(id));
        }
    }

    // --------------------------------------------------------------- gzip ---

    #[test]
    fn decodes_gzip_stream_without_normalized_buffer() {
        let Some(bytes) = corpus_bytes(KPAH_2008_GZIP) else {
            return;
        };
        let golden = golden(KPAH_2008_GZIP);
        assert_eq!(text(&golden["outer_compression"]), "gzip");

        let streamed = finished(builder_from_gzip_reader(bytes.as_slice()).unwrap());
        let buffered = decode(&bytes);

        assert_eq!(
            streamed.volume.provenance.compression.as_deref(),
            Some("gzip")
        );
        assert_same(&streamed.volume, &buffered.volume, "streamed vs buffered");
        assert_same(
            &read_gzip_volume_from_reader(bytes.as_slice()).unwrap(),
            &streamed.volume,
            "public streaming decode",
        );
        assert_volume_matches_golden(&streamed, &golden);
    }

    #[test]
    fn gzip_preview_waits_for_complete_displayable_cut() {
        let Some(bytes) = corpus_bytes(KTLX_1999_TRUNCATED_GZIP) else {
            return;
        };
        let golden = golden(KTLX_1999_TRUNCATED_GZIP);

        // The object ends 68 radials into its first cut: no cut completes.
        let preview = read_gzip_preview_from_bytes(&bytes, 1).unwrap();

        assert!(preview.is_none());
        let decoded = decode(&bytes);
        let volume = &decoded.volume;
        assert_eq!(volume.provenance.compression.as_deref(), Some("gzip"));
        assert_eq!(volume.sweeps.len(), 1);
        assert_eq!(volume.sweeps[0].nrays(), 68);
        assert_volume_matches_golden(&decoded, &golden);
    }

    #[test]
    fn gzip_preview_returns_completed_displayable_cut() {
        let Some(bytes) = corpus_bytes(KPAH_2008_GZIP) else {
            return;
        };
        let golden = golden(KPAH_2008_GZIP);
        let first = &golden["sweeps"][0];
        let radials = count(&first["radials"]);

        let preview = finished(
            builder_gzip_preview(&bytes, radials)
                .unwrap()
                .expect("completed first sweep preview"),
        );

        let volume = &preview.volume;
        assert_eq!(volume.attrs.instrument_name, "KPAH");
        assert_eq!(volume.provenance.compression.as_deref(), Some("gzip"));
        // The first cut ends with an end-of-elevation radial (status 2), so
        // the preview stops right there.
        assert_eq!(volume.sweeps.len(), 1);
        assert_sweep_matches_golden(
            &volume.sweeps[0],
            preview.statuses[0],
            first,
            "KPAH preview sweep 0",
        );
        let full = decode(&bytes).volume;
        assert_same(&volume.sweeps[0], &full.sweeps[0], "preview sweep 0");
        assert_same(
            &read_gzip_preview_from_bytes(&bytes, radials)
                .unwrap()
                .expect("public preview"),
            volume,
            "public preview",
        );

        // No cut of this volume has more radials than the first.
        assert!(
            list(&golden["sweeps"])
                .iter()
                .all(|sweep| count(&sweep["radials"]) <= radials)
        );
        assert!(
            read_gzip_preview_from_bytes(&bytes, radials + 1)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn gzip_preview_callback_continues_to_full_volume() {
        let Some(bytes) = corpus_bytes(KPAH_2008_GZIP) else {
            return;
        };
        let golden = golden(KPAH_2008_GZIP);
        let mut previews = Vec::new();

        let volume = read_gzip_volume_from_bytes_with_preview(&bytes, 1, |preview| {
            previews.push(preview);
        })
        .unwrap();

        assert_eq!(previews.len(), 1);
        assert_eq!(
            previews[0].provenance.decode.decoded_ray_count,
            count(&golden["sweeps"][0]["radials"])
        );
        let decoded =
            finished(builder_from_gzip_bytes_with_preview(&bytes, 1, |_| Ok(())).unwrap());
        assert_same(&volume, &decoded.volume, "volume after the preview");
        assert_same(&volume, &decode(&bytes).volume, "plain decode");
        assert_eq!(volume.provenance.decode.decoded_ray_count, 2520);
        assert_volume_matches_golden(&decoded, &golden);
    }

    // ------------------------------------------------------ LDM bzip2 ---

    #[test]
    fn decodes_bzip_blocks_without_concatenated_normalized_buffer() {
        let Some(bytes) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let golden = golden(KTLX_2024_TRIM);
        assert_eq!(list(&golden["ldm_records"]).len(), 9);

        let decoded = decode(&bytes);

        let volume = &decoded.volume;
        assert_eq!(
            volume.provenance.compression.as_deref(),
            Some("bzip2-blocks")
        );
        assert_eq!(volume.provenance.decode.decoded_ray_count, 960);
        assert_volume_matches_golden(&decoded, &golden);
    }

    #[test]
    fn bzip_preview_waits_for_complete_displayable_cut() {
        let (Some(start), Some(chunk)) =
            (corpus_bytes(KIWA_CHUNK_START), corpus_bytes(KIWA_CHUNK_002))
        else {
            return;
        };
        // A real archive prefix: the Start chunk (volume header + metadata
        // record) and the first 120 radials of a 720-radial cut.
        let mut bytes = start;
        bytes.extend_from_slice(&chunk);

        let preview = read_bzip_block_preview_from_bytes(&bytes, 1).unwrap();

        assert!(preview.is_none());
        let decoded = decode(&bytes);
        assert_eq!(decoded.volume.sweeps[0].nrays(), 120);
        assert_volume_matches_golden(&decoded, &golden(KIWA_CHUNKS_GOLDEN));
    }

    #[test]
    fn bzip_preview_returns_completed_displayable_cut() {
        let Some(bytes) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let trim_golden = golden(KTLX_2024_TRIM);
        let full = decode(&bytes).volume;

        // The trim keeps 480 of sweep 1's radials: the cut counts as complete
        // once the first radial of sweep 2 arrives.
        let preview = finished(
            builder_bzip_block_preview(&bytes, 1)
                .unwrap()
                .expect("completed first sweep preview"),
        );

        let volume = &preview.volume;
        assert_eq!(volume.attrs.instrument_name, "KTLX");
        assert_eq!(
            volume.provenance.compression.as_deref(),
            Some("bzip2-blocks")
        );
        assert_same(&volume.sweeps[0], &full.sweeps[0], "preview sweep 0");
        assert_sweep_matches_golden(
            &volume.sweeps[0],
            preview.statuses[0],
            &trim_golden["sweeps"][0],
            "trim preview",
        );
        assert_eq!(volume.sweeps.len(), 2);
        assert_eq!(volume.sweeps[1].nrays(), 1);
        assert_same(
            &read_bzip_block_preview_from_bytes(&bytes, 1)
                .unwrap()
                .expect("public preview"),
            volume,
            "public preview",
        );

        // The full volume's first cut ends with an end-of-elevation radial.
        let Some(bytes) = corpus_bytes(KTLX_2024_FULL) else {
            return;
        };
        let full_golden = golden(KTLX_2024_FULL);
        let preview = finished(
            builder_bzip_block_preview(&bytes, 1)
                .unwrap()
                .expect("completed first sweep preview"),
        );
        assert_eq!(preview.volume.sweeps.len(), 1);
        assert_eq!(preview.volume.provenance.decode.decoded_ray_count, 720);
        assert_sweep_matches_golden(
            &preview.volume.sweeps[0],
            preview.statuses[0],
            &full_golden["sweeps"][0],
            "full preview",
        );
    }

    #[test]
    fn bzip_preview_full_decode_reuses_path_and_returns_full_volume() {
        let Some(bytes) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let golden = golden(KTLX_2024_TRIM);
        let mut previews = Vec::new();

        let volume = read_volume_from_bytes_with_bzip_preview(&bytes, 1, |preview| {
            previews.push(preview);
        })
        .unwrap();

        assert_eq!(previews.len(), 1);
        assert_eq!(
            previews[0].sweeps[0].nrays(),
            count(&golden["sweeps"][0]["radials"])
        );
        let decoded = finished(builder_with_bzip_preview(&bytes, 1, |_| Ok(())).unwrap());
        assert_same(&volume, &decoded.volume, "volume after the preview");
        assert_same(&volume, &decode(&bytes).volume, "plain decode");
        assert_volume_matches_golden(&decoded, &golden);
    }

    #[test]
    fn multi_block_bzip_decode_matches_uncompressed_reference() {
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let golden = golden(KTLX_2024_TRIM);
        let expected_records = list(&golden["ldm_records"]);

        // Decompress each real LDM record on its own and frame the payloads
        // as an uncompressed Archive II file.
        let records = ldm_records(&file);
        assert_eq!(records.len(), expected_records.len());
        let mut uncompressed = file[..VOLUME_HEADER_LEN].to_vec();
        for ((offset, block), expected) in records.iter().zip(expected_records) {
            assert_eq!(*offset, count(&expected["offset"]));
            let payload = bzip2_decompress(block);
            assert_eq!(payload.len(), count(&expected["decompressed_len"]));
            uncompressed.extend_from_slice(&payload);
        }
        let reference = decode(&uncompressed).volume;
        assert_eq!(
            reference.provenance.compression.as_deref(),
            Some("uncompressed")
        );

        let decoded = decode(&file);

        let volume = &decoded.volume;
        assert_eq!(
            volume.provenance.compression.as_deref(),
            Some("bzip2-blocks")
        );
        let mut expected = reference.clone();
        expected.provenance.compression = volume.provenance.compression.clone();
        assert_same(volume, &expected, "block-bzip2 vs uncompressed decode");
        assert_volume_matches_golden(&decoded, &golden);

        // Re-block the same message stream every 1,000,003 bytes, so records
        // and messages are split across bzip2 blocks.
        let stream = &uncompressed[VOLUME_HEADER_LEN..];
        let chunks: Vec<&[u8]> = stream.chunks(1_000_003).collect();
        let mut reblocked = file[..VOLUME_HEADER_LEN].to_vec();
        for (index, chunk) in chunks.iter().enumerate() {
            let block = bzip2_compress(chunk);
            let len = i32::try_from(block.len()).unwrap();
            let control = if index + 1 == chunks.len() { -len } else { len };
            reblocked.extend_from_slice(&control.to_be_bytes());
            reblocked.extend_from_slice(&block);
        }
        assert_same(&decode(&reblocked).volume, volume, "re-blocked decode");
    }

    #[test]
    fn bzip_preview_fires_past_legacy_block_window() {
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let golden = golden(KTLX_2024_TRIM);
        let sweep_radials = count(&golden["sweeps"][0]["radials"]);
        let records = ldm_records(&file);
        assert_eq!(count(&golden["ldm_records"][1]["radials"]), 120);

        // Split record 1 (the first 120 radials of sweep 1) into 20 records
        // of 6 radials each and keep every other record byte for byte. Sweep
        // 1 then ends in the 24th record and sweep 2 starts in the 25th, past
        // the 16 records the old fixed preview scan window covered.
        let radials = bzip2_decompress(records[1].1);
        let bounds = message31_bounds(&radials);
        assert_eq!(bounds.len(), 121);
        let mut blocks = vec![records[0].1.to_vec()];
        for first in (0..120).step_by(6) {
            blocks.push(bzip2_compress(&radials[bounds[first]..bounds[first + 6]]));
        }
        blocks.extend(records[2..].iter().map(|(_, block)| block.to_vec()));
        assert_eq!(blocks.len(), 1 + 20 + 7);
        let mut reframed = file[..VOLUME_HEADER_LEN].to_vec();
        for (index, block) in blocks.iter().enumerate() {
            let len = i32::try_from(block.len()).unwrap();
            let control = if index + 1 == blocks.len() { -len } else { len };
            reframed.extend_from_slice(&control.to_be_bytes());
            reframed.extend_from_slice(block);
        }
        let original = decode(&file).volume;

        let mut previews = Vec::new();
        let volume =
            read_volume_from_bytes_with_bzip_preview(&reframed, sweep_radials, |preview| {
                previews.push(preview);
            })
            .unwrap();

        assert_eq!(previews.len(), 1);
        assert_eq!(previews[0].sweeps[0].nrays(), sweep_radials);
        assert_same(
            &previews[0].sweeps[0],
            &original.sweeps[0],
            "preview sweep 0",
        );
        assert_same(&volume, &original, "reframed decode");
    }

    #[test]
    fn corrupt_trailing_bzip_block_yields_partial_volume() {
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let golden = golden(KTLX_2024_TRIM);
        let clean = decode(&file).volume;

        // Zero the last LDM record's bzip2 data after its stream and block
        // magic ("BZh9" + "1AY&"), so it still frames as bzip2 but fails.
        let (offset, block) = *ldm_records(&file).last().unwrap();
        let start = offset + 4;
        let mut corrupt = file.clone();
        corrupt[start + 8..start + block.len()].fill(0);

        let volume = decode(&corrupt).volume;

        // What MetPy reads from the file without that record: 480 + 360.
        let expected: Vec<usize> =
            list(&golden["layout_checks"]["without_last_record_sweep_radials"])
                .iter()
                .map(count)
                .collect();
        assert_eq!(expected, vec![480, 360]);
        let radials: Vec<usize> = volume.sweeps.iter().map(Sweep::nrays).collect();
        assert_eq!(radials, expected);
        assert_eq!(volume.provenance.decode.decoded_ray_count, 840);
        assert_same(&volume.sweeps[0], &clean.sweeps[0], "sweep 0");
        assert_eq!(
            volume.provenance.decode.skipped_message_count,
            clean.provenance.decode.skipped_message_count + 1
        );
    }

    #[test]
    fn corrupt_first_bzip_block_is_a_hard_error() {
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        // Zero the metadata record's bzip2 data after its magic.
        let (offset, block) = ldm_records(&file)[0];
        let start = offset + 4;
        let mut corrupt = file.clone();
        corrupt[start + 8..start + block.len()].fill(0);

        assert!(matches!(
            read_volume_from_bytes(&corrupt),
            Err(NexradError::Compression(_))
        ));
        assert!(read_bzip_block_preview_from_bytes(&corrupt, 1).is_err());
    }

    #[test]
    fn corrupt_bzip_block_error_names_the_record_path() {
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        // The metadata record's bzip2 data zeroed after its stream and block
        // magic, decoded on its own: the error names the record path.
        let (_, block) = ldm_records(&file)[0];
        let mut bad = block.to_vec();
        bad[8..].fill(0);
        let mut decoded = Vec::new();
        let error = decompress_bzip_block_into(&bad, &mut decoded).unwrap_err();
        assert!(
            matches!(&error, NexradError::Compression(reason) if reason.starts_with("block-bzip chunk: bzip2:")),
            "unexpected error: {error}"
        );
        assert!(decoded.is_empty(), "output restored on error");
    }

    /// The metadata record of a whole-file bzip2 volume is decoded from the
    /// stream's first blocks only (backlog: a bounded early stop instead of
    /// expanding the whole stream): with 100 kB blocks it equals the
    /// uncompressed record, and damage in the last quarter of the stream,
    /// which fails the volume decode, leaves the metadata record readable.
    #[test]
    fn whole_file_bzip2_metadata_record_decodes_only_its_blocks() {
        use std::io::Write;
        let Some(file) = corpus_bytes(KTLX_2013_TRIM) else {
            return;
        };
        let (bytes, _) = normalize_archive_bytes(&file).unwrap();
        let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(1));
        encoder.write_all(&bytes).unwrap();
        let compressed = encoder.finish().unwrap();
        let record_len = 134 * RECORD_BYTES;
        assert!(bytes.len() > VOLUME_HEADER_LEN + 3 * record_len);

        let record = messages::metadata_record(&compressed).unwrap();
        assert!(*record == bytes[VOLUME_HEADER_LEN..VOLUME_HEADER_LEN + record_len]);

        let mut damaged = compressed.clone();
        let len = damaged.len();
        damaged[len * 7 / 8] ^= 0x55;
        assert!(normalize_archive_bytes(&damaged).is_err());
        assert!(read_volume_from_bytes(&damaged).is_err());
        let record = messages::metadata_record(&damaged).unwrap();
        assert!(*record == bytes[VOLUME_HEADER_LEN..VOLUME_HEADER_LEN + record_len]);
    }

    /// A whole-file bzip2 wrapper around LDM records: the metadata record
    /// is the first LDM record, taken from the stream's first blocks, which
    /// are decoded once. Damage or truncation past those blocks, which fails
    /// the volume decode, leaves it readable; damage inside them fails with
    /// the metadata record's context, as the whole decode did before.
    #[test]
    fn whole_file_bzip2_of_ldm_records_reads_the_first_record_from_its_blocks() {
        use std::io::Write;
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let expected = messages::metadata_record(&file).unwrap().into_owned();
        let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(1));
        encoder.write_all(&file).unwrap();
        let compressed = encoder.finish().unwrap();
        // 100 kB blocks: the metadata prefix (325,936 bytes) is the first
        // four of eight.
        assert!(file.len() > 7 * 100_000, "{} bytes", file.len());
        let len = compressed.len();

        assert!(*messages::metadata_record(&compressed).unwrap() == expected[..]);

        let mut damaged = compressed.clone();
        damaged[len * 7 / 8] ^= 0x55;
        assert!(read_volume_from_bytes(&damaged).is_err());
        assert!(*messages::metadata_record(&damaged).unwrap() == expected[..]);
        let truncated = &compressed[..len * 6 / 10];
        assert!(read_volume_from_bytes(truncated).is_err());
        assert!(*messages::metadata_record(truncated).unwrap() == expected[..]);

        let mut early = compressed.clone();
        early[64] ^= 0x55;
        let error = messages::metadata_record(&early).unwrap_err();
        assert!(
            matches!(&error, NexradError::Compression(reason)
                if reason.starts_with("whole-file compressed Level II metadata record: bzip2:")),
            "{error}"
        );
    }

    #[test]
    fn whole_file_bzip2_archive_decodes_like_the_uncompressed_bytes() {
        // No corpus file is a whole-file bzip2 stream; the KTLX 2013 trim's
        // decompressed bytes (header, metadata and radial messages) wrapped
        // in one bzip2 stream stand in for one.
        let Some(file) = corpus_bytes(KTLX_2013_TRIM) else {
            return;
        };
        let (bytes, _) = normalize_archive_bytes(&file).unwrap();
        let compressed = bzip2_compress(&bytes);
        assert!(compressed.starts_with(b"BZh"));

        let (normalized, compression) = normalize_archive_bytes(&compressed).unwrap();
        assert_eq!(compression, ArchiveCompression::Bzip2WholeFile);
        assert_eq!(normalized, bytes);

        let mut expected = decode(&bytes).volume;
        assert_eq!(
            expected.provenance.compression.as_deref(),
            Some("uncompressed")
        );
        expected.provenance.compression = Some("bzip2-whole-file".to_owned());
        let decoded = decode(&compressed);
        assert_same(&decoded.volume, &expected, "whole-file bzip2 decode");
        assert_volume_matches_golden(&decoded, &golden(KTLX_2013_TRIM));

        // Bytes after the end-of-stream marker are ignored, as before.
        let mut trailing = compressed.clone();
        trailing.extend_from_slice(&[0u8; 64]);
        assert_same(
            &decode(&trailing).volume,
            &expected,
            "trailing bytes ignored",
        );
    }

    /// A gzip or whole-file bzip2 copy of an LDM-compressed file (the KTLX
    /// 2024 trim wrapped whole) decodes as the unwrapped file does, through
    /// every entry point, with the wrapper as its compression. Before, the
    /// expanded LDM control words and bzip2 bytes were parsed as messages
    /// and a few garbage radials came back without an error.
    #[test]
    fn ldm_records_inside_a_whole_file_wrapper_decode_like_the_unwrapped_file() {
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let golden = golden(KTLX_2024_TRIM);
        let unwrapped = decode(&file).volume;
        assert_eq!(
            unwrapped.provenance.compression.as_deref(),
            Some("bzip2-blocks")
        );
        let unwrapped_metadata = read_volume_with_metadata(&file).unwrap().metadata;
        assert!(
            unwrapped_metadata.errors.is_empty(),
            "{:?}",
            unwrapped_metadata.errors
        );
        let (normalized, _) = normalize_archive_bytes(&file).unwrap();

        let mut gzip_encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gzip_encoder.write_all(&file).unwrap();
        let gzipped = gzip_encoder.finish().unwrap();
        let bzipped = bzip2_compress(&file);

        for (wrapped, compression) in [
            (&gzipped, ArchiveCompression::Gzip),
            (&bzipped, ArchiveCompression::Bzip2WholeFile),
        ] {
            let name = compression.as_str();
            let mut expected = unwrapped.clone();
            expected.provenance.compression = Some(name.to_owned());

            let decoded = decode(wrapped);
            assert_same(&decoded.volume, &expected, name);
            assert_volume_matches_golden(&decoded, &golden);

            let with_metadata = read_volume_with_metadata(wrapped).unwrap();
            assert_same(&with_metadata.volume, &expected, name);
            assert_eq!(
                with_metadata.metadata, unwrapped_metadata,
                "{name} metadata"
            );

            let (wrapped_normalized, wrapped_compression) =
                normalize_archive_bytes(wrapped).unwrap();
            assert_eq!(wrapped_compression, compression);
            assert!(wrapped_normalized == normalized, "{name} normalized bytes");

            // What the router does with a gzip file: expand it to sniff the
            // format, then parse the expansion.
            let expanded = messages::record_bytes(wrapped).unwrap();
            let mut with_header = file[..VOLUME_HEADER_LEN].to_vec();
            with_header.extend_from_slice(&expanded);
            assert!(with_header == normalized, "{name} record bytes");
            let mut inner = Vec::new();
            if compression == ArchiveCompression::Gzip {
                inner =
                    gzip::inflate_gzip_members_limited(wrapped, MAX_DECODED_RADAR_BYTES, "test")
                        .unwrap();
            } else {
                decompress_bzip2_stream_into(wrapped, &mut inner, MAX_DECODED_RADAR_BYTES, "test")
                    .unwrap();
            }
            assert!(inner == file);
            let from_expansion = read_normalized_volume_bytes(&inner, compression).unwrap();
            assert_same(&from_expansion, &expected, name);
        }

        let mut expected = unwrapped.clone();
        expected.provenance.compression = Some("gzip".to_owned());
        let streamed = read_gzip_volume_from_reader(&gzipped[..]).unwrap();
        assert_same(&streamed, &expected, "gzip reader");

        let mut previews = Vec::new();
        let full = read_gzip_volume_from_bytes_with_preview(&gzipped, 100, |preview| {
            previews.push(preview);
        })
        .unwrap();
        assert_same(&full, &expected, "gzip decode with preview");
        assert_eq!(previews.len(), 1);
        assert_same(
            &previews[0].sweeps[0],
            &expected.sweeps[0],
            "preview sweep 0",
        );
        let preview = read_gzip_preview_from_bytes(&gzipped, 100)
            .unwrap()
            .unwrap();
        assert_same(
            &preview.sweeps[0],
            &expected.sweeps[0],
            "gzip preview sweep 0",
        );
    }

    /// An LDM-compressed file cut short inside a record decodes to the
    /// records before the cut, as MetPy reads the file without its last
    /// record (the golden's 480 + 360 radials), with the cut record counted
    /// as skipped, as a damaged record is; its metadata record is still
    /// read. A record that is not a bzip2 stream decodes the same way. A cut
    /// inside the first record is a truncation error. Before, all three
    /// were parsed as uncompressed messages: a few garbage radials, or an
    /// empty volume, without an error.
    #[test]
    fn ldm_file_cut_short_decodes_the_records_before_the_cut() {
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let golden = golden(KTLX_2024_TRIM);
        let expected: Vec<usize> =
            list(&golden["layout_checks"]["without_last_record_sweep_radials"])
                .iter()
                .map(count)
                .collect();
        let clean = decode(&file).volume;
        let (offset, block) = *ldm_records(&file).last().unwrap();

        let cut = &file[..offset + 4 + block.len() / 2];
        let mut not_bzip2 = file.clone();
        not_bzip2[offset + 4..offset + 7].fill(0);
        for (bytes, what) in [(cut, "cut short"), (&not_bzip2[..], "not bzip2")] {
            let volume = decode(bytes).volume;
            let radials: Vec<usize> = volume.sweeps.iter().map(Sweep::nrays).collect();
            assert_eq!(radials, expected, "{what}");
            assert_same(&volume.sweeps[0], &clean.sweeps[0], what);
            assert_eq!(
                volume.provenance.decode.skipped_message_count,
                clean.provenance.decode.skipped_message_count + 1,
                "{what}"
            );
            let preview = read_bzip_block_preview_from_bytes(bytes, 100)
                .unwrap()
                .unwrap();
            assert_same(&preview.sweeps[0], &clean.sweeps[0], what);
            let metadata = read_volume_with_metadata(bytes).unwrap().metadata;
            assert!(metadata.errors.is_empty(), "{what}: {:?}", metadata.errors);
            assert!(metadata.vcp.is_some(), "{what}");
        }

        let first = &file[..VOLUME_HEADER_LEN + 4 + 1000];
        for result in [
            read_volume_from_bytes(first).map(|_| ()),
            read_bzip_block_preview_from_bytes(first, 1).map(|_| ()),
            normalize_archive_bytes(first).map(|_| ()),
        ] {
            assert!(
                matches!(
                    result,
                    Err(NexradError::Truncated {
                        what: "LDM compressed record",
                        offset: VOLUME_HEADER_LEN,
                        available: 1000,
                        ..
                    })
                ),
                "{result:?}"
            );
        }
    }

    #[test]
    fn bzip2_stream_output_limit_is_exact_and_restores_the_buffer() {
        // The KTLX 2024 trim's first radial record: a real LDM bzip2 stream
        // whose decoded size is the exact limit.
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let (_, compressed) = ldm_records(&file)[1];
        let payload = bzip2_decompress(compressed);
        assert!(
            payload.len() > 100_000,
            "real record: {} bytes",
            payload.len()
        );
        let context = "limit test";

        let mut output = b"kept".to_vec();
        decompress_bzip2_stream_into(compressed, &mut output, payload.len(), context).unwrap();
        assert_eq!(&output[..4], b"kept");
        assert_eq!(&output[4..], &payload[..]);

        let mut output = b"kept".to_vec();
        let error =
            decompress_bzip2_stream_into(compressed, &mut output, payload.len() - 1, context)
                .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "unsupported or corrupt compression wrapper: {context} expands beyond the {}-byte limit",
                payload.len() - 1
            )
        );
        assert_eq!(output, b"kept", "output restored to its original length");
    }

    #[test]
    fn oversized_bzip_block_is_rejected_at_the_per_block_limit() {
        // A real radial record's decoded bytes repeated past the 16 MiB block
        // cap and recompressed: the record's own messages, only more of them.
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let radials = bzip2_decompress(ldm_records(&file)[1].1);
        let mut payload = Vec::with_capacity(MAX_BZIP_BLOCK_DECODED_BYTES + radials.len());
        while payload.len() <= MAX_BZIP_BLOCK_DECODED_BYTES {
            payload.extend_from_slice(&radials);
        }
        let compressed = bzip2_compress(&payload);
        let mut decoded = Vec::new();
        let error = decompress_bzip_block_into(&compressed, &mut decoded).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "unsupported or corrupt compression wrapper: block-bzip chunk expands beyond the {MAX_BZIP_BLOCK_DECODED_BYTES}-byte limit"
            )
        );
        assert!(decoded.is_empty());
    }

    #[test]
    fn bzip_buffer_pool_keeps_only_block_sized_buffers() {
        // Too small and too large buffers are dropped, never pooled. The pool
        // is process-wide and other tests use it concurrently, so only the
        // invariants are checked, not the exact contents.
        recycle_bzip_buffer(Vec::with_capacity(BZIP_BUFFER_POOL_MIN_CAPACITY - 1));
        recycle_bzip_buffer(Vec::with_capacity(BZIP_BUFFER_POOL_MAX_CAPACITY + 1));
        recycle_bzip_buffer(Vec::with_capacity(BZIP_BUFFER_POOL_MIN_CAPACITY));
        {
            let pool = BZIP_BUFFER_POOL.lock().unwrap();
            assert!(pool.buffers.len() <= BZIP_BUFFER_POOL_MAX_BUFFERS);
            assert!(pool.retained_bytes <= BZIP_BUFFER_POOL_MAX_BYTES);
            assert_eq!(
                pool.retained_bytes,
                pool.buffers.iter().map(Vec::capacity).sum::<usize>()
            );
            for buffer in &pool.buffers {
                assert!(buffer.is_empty());
                assert!(
                    (BZIP_BUFFER_POOL_MIN_CAPACITY..=BZIP_BUFFER_POOL_MAX_CAPACITY)
                        .contains(&buffer.capacity())
                );
            }
        }
        let taken = take_bzip_buffer();
        assert!(taken.is_empty());
        assert!(taken.capacity() == 0 || taken.capacity() >= BZIP_BUFFER_POOL_MIN_CAPACITY);
    }

    /// Committed KIWA volume 307 Start chunk plus its first two intermediate
    /// chunks, decompressed (240 real Message 31 radials).
    fn kiwa_chunk_prefix_normalized() -> Vec<u8> {
        let mut raw = Vec::new();
        for id in [KIWA_CHUNK_START, KIWA_CHUNK_002, KIWA_CHUNK_003] {
            let path = recast_radar_testdata::path(id).unwrap_or_else(|e| panic!("{e}"));
            raw.extend(fs::read(path).expect("read committed chunk"));
        }
        normalize_archive_bytes(&raw)
            .expect("real chunks decompress")
            .0
    }

    #[test]
    fn real_volume_exceeding_the_output_budget_is_rejected() {
        let bytes = kiwa_chunk_prefix_normalized();
        let volume = builder_from_normalized(
            &bytes,
            ArchiveCompression::Bzip2Blocks,
            DecodeBudget::volume(),
        )
        .expect("real chunks fit the default budget")
        .finish()
        .unwrap()
        .0;
        let needed = bounded_read::volume_field_capacity_bytes(&volume);
        assert!(
            needed > 1024 * 1024,
            "240 super-resolution radials: {needed}"
        );

        let error = builder_from_normalized(
            &bytes,
            ArchiveCompression::Bzip2Blocks,
            DecodeBudget::new(needed / 2),
        )
        .expect_err("half the needed budget must fail");
        assert!(
            matches!(&error, NexradError::LimitExceeded(reason) if reason.contains("limit")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn real_chunks_decode_natively_with_vcp_fixed_angles() {
        // The Start chunk carries the metadata record (Message 5), so the
        // sweep's fixed angle is the VCP cut angle rather than the opening
        // radial's elevation.
        let bytes = kiwa_chunk_prefix_normalized();
        let volume = read_normalized_volume_bytes(&bytes, ArchiveCompression::Bzip2Blocks).unwrap();
        assert_eq!(volume.sweeps.len(), 1);
        let sweep = &volume.sweeps[0];
        assert_eq!(sweep.nrays(), 240);
        assert_eq!(volume.scan.vcp_pattern, Some(215));
        assert_eq!(volume.scan.name.as_deref(), Some("VCP-215"));
        assert!(
            (sweep.fixed_angle_deg - 0.5).abs() < 0.05,
            "fixed angle {}",
            sweep.fixed_angle_deg
        );
        assert_ne!(sweep.fixed_angle_deg, sweep.rays.elevation_deg[0]);
        assert!(sweep.rays.time_s.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(sweep.range.spacing_m(), Some(250.0));
        let reflectivity = field(&volume, 0, &FieldName::Dbzh);
        assert_eq!(reflectivity.nrays, 240);
        assert!(reflectivity.absent_rows.is_empty());
        assert_eq!(volume.time_coverage.map(|c| c.start), volume.ray_time(0, 0));
    }

    #[test]
    fn pipelined_decode_works_on_single_thread_rayon_pool() {
        let Some(bytes) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let golden = golden(KTLX_2024_TRIM);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let decoded = pool.install(|| decode(&bytes));

        assert_same(
            &decoded.volume,
            &decode(&bytes).volume,
            "single-thread decode",
        );
        assert_volume_matches_golden(&decoded, &golden);
    }

    // ----------------------------------------------------------- layouts ---

    #[test]
    fn expected_radials_follow_message31_azimuth_resolution_code() {
        assert_eq!(expected_radials_for_azimuth_resolution(1), 720);
        assert_eq!(expected_radials_for_azimuth_resolution(2), 360);
        assert_eq!(
            expected_radials_for_azimuth_resolution(0),
            FALLBACK_RADIALS_PER_CUT
        );
    }

    #[test]
    fn decodes_gr2_style_variable_framed_msg31_records() {
        // GR2 ".msg31" exports keep the AR2V volume header but carry only a
        // few fixed-size metadata records, then Message 31 records back to
        // back, uncompressed. The corpus has no GR2 export, so the layout is
        // cut from the real KTLX 2024 trim: its volume header, its Message 2
        // and 5 records, then all its Message 31 records. MetPy reads that
        // layout as the same two sweeps (golden `layout_checks`).
        let Some(file) = corpus_bytes(KTLX_2024_TRIM) else {
            return;
        };
        let golden = golden(KTLX_2024_TRIM);
        let checks = &golden["layout_checks"];
        let keep: Vec<u8> = list(&checks["gr2_metadata_types"])
            .iter()
            .map(|t| u8::try_from(int(t)).unwrap())
            .collect();
        let (normalized, _) = normalize_archive_bytes(&file).unwrap();
        let mut gr2 = normalized[..VOLUME_HEADER_LEN].to_vec();
        let mut cursor = VOLUME_HEADER_LEN;
        loop {
            let header = parse_message_header(&normalized, cursor + CONTROL_WORD_LEN).unwrap();
            if header.message_type == 31 {
                break;
            }
            if header.size_halfwords != 0 && keep.contains(&header.message_type) {
                gr2.extend_from_slice(&normalized[cursor..cursor + RECORD_BYTES]);
            }
            cursor += RECORD_BYTES;
        }
        message31_bounds(&normalized[cursor..]);
        gr2.extend_from_slice(&normalized[cursor..]);
        assert_eq!(gr2.len(), count(&checks["gr2_len"]));
        let original = decode(&file);

        let decoded = decode(&gr2);

        let volume = &decoded.volume;
        assert_eq!(
            volume.provenance.compression.as_deref(),
            Some("uncompressed")
        );
        let radials: Vec<usize> = volume.sweeps.iter().map(Sweep::nrays).collect();
        let expected: Vec<usize> = list(&checks["gr2_sweep_radials"])
            .iter()
            .map(count)
            .collect();
        assert_eq!(radials, expected);
        // The layout keeps no Message 18, so its sweeps have no transmitter
        // pulse width; everything else is the original's.
        let mut original_sweeps = original.volume.sweeps.clone();
        for sweep in &mut original_sweeps {
            assert!(sweep.ray_vars.pulse_width_s.take().is_some());
        }
        assert_same(&volume.sweeps, &original_sweeps, "sweeps");
        assert_eq!(
            volume.attrs.instrument_name,
            original.volume.attrs.instrument_name
        );
        assert_eq!(volume.location, original.volume.location);
        assert_eq!(volume.scan, original.volume.scan);
        assert_eq!(decoded.header_time, original.header_time);
        assert_eq!(volume.time_reference, original.volume.time_reference);
        assert_eq!(volume.provenance.decode.message_count, keep.len() + 960);

        // GR2 exports write a nonstandard volume header date: with the date
        // and time zeroed, the volume time comes from the first radial.
        gr2[12..20].fill(0);
        let decoded = decode(&gr2);
        assert_eq!(
            decoded.header_time.timestamp_millis(),
            int(&golden["sweeps"][0]["first_epoch_ms"])
        );
        assert_same(
            &decoded.volume.sweeps,
            &original_sweeps,
            "sweeps with a zeroed header date",
        );
    }
}
