//! NEXRAD Archive II (Level II) writer: any FM301 [`Volume`] to `AR2V0006`
//! bytes that the ICD 2620010 / 2620002 readers accept.
//!
//! See `docs/level2/writer.md` for the full description, the mapping rules
//! and how the output was verified.
//!
//! # Output
//!
//! - A 24-byte volume header (`AR2V0006.NNN`, the volume start date and time,
//!   the 4-character site identifier).
//! - The 134-frame metadata record of real Archive II files: Message 18 (RDA
//!   adaptation data: site name, location, frequency, antenna gain, beam
//!   width) in frames 127 to 130, Message 5 (the volume coverage pattern,
//!   synthesised from the sweeps: one cut per sweep) in frame 133 and
//!   Message 2 (RDA status) in frame 134; the other frames are empty. When the
//!   source is itself Level II, its own metadata record can be carried over
//!   instead, made to agree with the written volume ([`SourceMetadata`]).
//! - One Message 31 per radial (Data Header Block, VOL, ELV and RAD constant
//!   blocks, then the REF, VEL, SW, ZDR, PHI, RHO and CFP moments the ray
//!   has), each cut's radials in the order they were collected (an ODIM
//!   sweep stored from north is written from its earliest ray,
//!   [`WrittenRays`]), in records of
//!   [`WriteOptions::radials_per_record`] radials (120, as NOAA's files have
//!   them) that run on across elevation cuts or end with each cut
//!   ([`RecordLayout`]), with a Level II source's mid-volume non-radial
//!   messages among them where the source had them.
//! - Records either uncompressed or as LDM records (a 4-byte control word,
//!   negative for the last record, then one bzip2 stream per record), and
//!   optionally a gzip wrapper around the whole file.
//!
//! [`realtime`] cuts the same records into real-time chunks (`S`, `I`, `E`
//! files as in the AWS chunks bucket), for a complete volume or while its
//! sweeps arrive ([`realtime::ChunkWriter`]), and [`polling`] publishes
//! volumes into a GR2Analyst polling directory (`dir.list` with
//! `<size> <filename>` lines) following the GRLevelX polling conventions.
//!
//! # Refusals
//!
//! Everything is checked before the first byte is produced: an error means
//! nothing was written (apart from I/O errors of the sink). Level II cannot
//! hold RHI or other non-PPI sweeps, more than 32 elevation cuts, gates that
//! do not start at 0 to 32767 m with a whole-metre spacing (within
//! [`WriteOptions::max_range_error_m`]), more than 16384 gates, radials over
//! 65535 bytes, non-finite angles or times before 1970; each is a typed
//! [`WriteError`], as are per-ray arrays that do not match the ray count
//! (a volume changed without
//! [`Sweep::seal`](recast_radar_core::model::Sweep::seal)).
//!
//! A Level II file holds one volume scan. A volume whose sweeps come from
//! more than one scan cycle ([`scan_cycles`](recast_radar_core::model::scan_cycles):
//! a cut collected again, a pause of minutes, or a second start of a Level
//! II volume) is refused ([`WriteError::MixedScanCycles`]);
//! [`split_scan_cycles`](recast_radar_core::model::split_scan_cycles) gives
//! one volume per cycle to write on its own. A foreign volume's cuts are
//! written in the order their sweeps were collected
//! ([`WriteSummary::written_sweeps`]), as Level II holds radials; a Level II
//! source keeps the order it is given. The volume header time is the
//! earliest written radial's.
//!
//! Every radial carries at least one moment: Py-ART and xradar take a cut's
//! moments from its first radial and fail on one without any. A ray on which
//! no written moment has data is left out, and a sweep left without rays (no
//! field maps to a moment, or every mapped field was left out) is left out
//! too; both are reported in the [`WriteSummary`].

#![deny(missing_docs)]

mod compress;
mod encode;
mod plan;
pub mod polling;
mod quantize;
pub mod realtime;

use std::io::Write;

use chrono::{DateTime, Utc};
use recast_radar_core::model::{CycleBreak, FieldName, Volume};
use thiserror::Error;

use crate::metadata::NexradMetadata;

pub use plan::Moment;

/// Record compression of the written file.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Compression {
    /// Records stored as they are: the volume header, then the 134 metadata
    /// frames, then the Message 31 frames.
    None,
    /// LDM records, as NOAA distributes Archive II: a 4-byte big-endian
    /// control word holding the compressed size (negative for the last
    /// record), then one bzip2 stream per record.
    #[default]
    Bzip2LdmRecords,
}

/// How radials are grouped into records (the LDM records of
/// [`Compression::Bzip2LdmRecords`], and the real-time chunks).
///
/// NOAA's cuts are 360 or 720 radials, so their 120-radial records end with
/// the cut under either layout; the two differ only for cuts of other sizes
/// (JMA's 512 radials, pre-2008 NEXRAD, many foreign radars).
///
/// Neither lets xradar 0.12 read a compressed file with such cuts: it
/// addresses the messages of the data records as if each record held 120,
/// and reads a sweep right only when the sweep starts a record. It reads
/// [`Compression::None`] files in full whatever the cuts
/// (`docs/level2/writer.md`, "Record layout").
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum RecordLayout {
    /// Records of [`WriteOptions::radials_per_record`] radials that run on
    /// across elevation cuts; only the volume's last record holds fewer.
    /// xradar 0.12 lists every sweep of such a file but fails on the data of
    /// a sweep that starts inside a record.
    #[default]
    Continuous,
    /// Records that end with their elevation cut: a cut's last record holds
    /// the rest of its radials, so each real-time chunk holds radials of one
    /// cut and a cut's last chunk goes out as soon as the cut is complete.
    /// xradar 0.12 reads only the first sweep of such a file when a cut is
    /// not a multiple of `radials_per_record` radials.
    WithinCuts,
}

/// How field values become Message 31 gate codes.
///
/// Under every policy, fields that are already NEXRAD-coded (decoded from
/// Level II) keep their codes, and each moment gets one coding for the whole
/// volume (Py-ART decodes every sweep of a moment with the first sweep's
/// scale and offset).
///
/// The default, [`Quantization::Standard`], writes the codings NOAA's
/// current files carry, which every Level II reader (GR2Analyst among them)
/// expects: PHI in codes up to 1023, REF, VEL and SW in 8 bits.
/// [`Quantization::Precise`] never codes a value more coarsely than its
/// source stores it, which for 16-bit and float sources means 16-bit moments
/// and PHI codes up to 65535 that readers keeping only NEXRAD's bits misread
/// (see `docs/level2/writer.md`, "Quantisation").
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Quantization {
    /// No worse than the source. The ICD's typical coding when every value
    /// lies on it, else an exact coding of the values' own evenly spaced
    /// grid (8-bit words when it has at most 254 levels, else 16-bit), else
    /// the finest 16-bit coding that covers every value. 16-bit and float
    /// sources keep their precision. Py-ART, MetPy, RSL, LROSE Radx, the
    /// `nexrad` crate and this decoder read these 16-bit moments (Py-ART's
    /// own limits, such as a site identifier starting with `T`, are listed
    /// in `docs/level2/writer.md`). xradar 0.12 keeps only the low 8 bits of
    /// 16-bit moments other than ZDR and PHI, and the low 11 and 10 bits of
    /// those two, so it misreads the 16-bit moments this policy writes for
    /// 16-bit and float sources, as do readers that keep only NEXRAD's PHI
    /// bits (codes above 1023). Whether GR2Analyst reads a 16-bit REF, VEL
    /// or SW has not been checked.
    Precise,
    /// Every moment in the word size NEXRAD files use: REF, VEL, SW, RHO and
    /// CFP in 8 bits, ZDR and PHI in 8 bits or in 16 bits up to codes 2047
    /// and 1023 (what current radars send, and all xradar 0.12 reads). The
    /// coding is the ICD's typical one when every value lies on it, else an
    /// exact coding of the values' own evenly spaced grid when it fits those
    /// codes (lossless for 8-bit sources), else the finest coding in those
    /// codes that covers every value: coarser than a 16-bit or float source
    /// ([`MomentReport::max_abs_error`] says by how much).
    Compatible,
    /// The ICD's typical coding of every moment, as NOAA's current files
    /// carry it (REF 8-bit at scale 2 and offset 66, VEL and SW 8-bit at 2
    /// and 129, ZDR 16-bit at 32 and 418, PHI 16-bit at 2.8361 and 2, RHO
    /// 8-bit at 300 and -60.5, CFP 8-bit at 1 and 8), rounding each value to
    /// the nearest code. It never clips: a moment with a value outside its
    /// typical coding's range is coded as [`Quantization::Compatible`]
    /// codes it instead. The default.
    #[default]
    Standard,
}

/// Options of the Level II writer. Start from [`WriteOptions::default`] and
/// set the fields you need.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct WriteOptions {
    /// Record compression (default: LDM bzip2 records).
    pub compression: Compression,
    /// Wrap the whole file in gzip (default: `false`).
    pub gzip: bool,
    /// Site identifier written in the volume header and every radial: 1 to 4
    /// characters from `[A-Za-z0-9_]`, padded with `_` to 4. `None` derives
    /// it from `Volume::attrs.instrument_name` (see `docs/level2/writer.md`).
    /// Py-ART 2.3 takes every site starting with `T` for a TDWR and cannot
    /// open a file whose `T` site is missing from its station table (JMA
    /// Takayasu derives as `TAKA`); set another identifier
    /// where Py-ART must read the file.
    pub icao: Option<String>,
    /// VCP number for the VOL blocks and the synthesised Messages 2 and 5.
    /// `None` keeps `Volume::scan.vcp_pattern`, else 0 (no pattern).
    pub vcp: Option<u16>,
    /// Value coding policy.
    pub quantization: Quantization,
    /// Radials per record (1 to 65535). Real files use 120, and xradar 0.12
    /// reads compressed files only when every record but a volume's last
    /// holds 120 messages and every sweep starts a record ([`RecordLayout`]).
    pub radials_per_record: usize,
    /// Whether records run on across elevation cuts (the default) or end
    /// with each cut.
    pub record_layout: RecordLayout,
    /// Volume number of the header's `AR2V0006.NNN` extension (1 to 999).
    /// `None` keeps the number of a Level II source, else 1.
    pub volume_number: Option<u16>,
    /// Largest range error, in metres at the last gate, accepted when a
    /// field's first gate or spacing is not a whole number of metres. `None`
    /// accepts up to half of one gate spacing.
    pub max_range_error_m: Option<f64>,
    /// Leave out gates whose centre lies before the radar (negative range),
    /// which Message 31 cannot hold; `false` (the default) refuses such
    /// fields. Message 1 volumes place their Doppler gates from -375 m.
    pub drop_negative_range_gates: bool,
    /// Explicit field-to-moment assignments, applied before the automatic
    /// mapping (for example a DORADE `DB_DBZ2` field to [`Moment::Ref`]).
    pub field_map: Vec<(FieldName, Moment)>,
    /// Write the cuts in the order of `Volume::sweeps` instead of the order
    /// they were collected (the default for volumes not decoded from Level
    /// II): for a volume put in order of elevation, lowest first, whose
    /// radar scans from the top down (ECCC). Rays within a cut are still
    /// written from the earliest collected.
    pub keep_sweep_order: bool,
    /// Nyquist velocity (m/s) written in the RAD block of every radial
    /// whose source has none (`Sweep::ray_vars.nyquist_velocity_mps` absent,
    /// not finite or not positive); `None` writes 0 there, which readers
    /// take as unknown. The JMA decoder leaves it unset (staggered PRF). The
    /// writer never invents one: set this only to the radar's own value
    /// (for example from its PRF). More than 0 and at most 327.67 m/s (the
    /// RAD block's 0.01 m/s steps).
    pub nyquist_velocity_mps: Option<f32>,
    /// Unambiguous range (m) written in the RAD block of every radial whose
    /// source has none (`Sweep::ray_vars.unambiguous_range_m` absent, not
    /// finite or not positive); `None` writes 0, which readers take as
    /// unknown. Set it only to the radar's own value. More than 0 and at
    /// most 3276.7 km (the RAD block's 0.1 km steps).
    pub unambiguous_range_m: Option<f32>,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            compression: Compression::default(),
            gzip: false,
            icao: None,
            vcp: None,
            quantization: Quantization::default(),
            radials_per_record: 120,
            record_layout: RecordLayout::default(),
            volume_number: None,
            max_range_error_m: None,
            drop_negative_range_gates: false,
            field_map: Vec::new(),
            keep_sweep_order: false,
            nyquist_velocity_mps: None,
            unambiguous_range_m: None,
        }
    }
}

/// Level II context carried over from a source that was itself Level II,
/// for a faithful re-encoding. Every part is optional.
#[derive(Clone, Copy, Debug, Default)]
pub struct SourceMetadata<'a> {
    /// The source's decoded metadata ([`crate::read_volume_with_metadata`]):
    /// its per-sweep VOL, ELV and RAD blocks and its volume header time are
    /// reused.
    pub metadata: Option<&'a NexradMetadata>,
    /// The source's decompressed metadata record
    /// ([`crate::messages::metadata_record`]), written verbatim in place of
    /// the synthesised Messages 2, 5 and 18. It is ignored (and reported in
    /// [`WriteSummary::notes`]) when it holds radials, as records of files
    /// from before 2005 do.
    pub metadata_record: Option<&'a [u8]>,
    /// The non-radial messages of the source's data records
    /// ([`data_messages`]): RDA status updates the RDA sent mid-volume,
    /// written again among the radials where the source had them. Empty:
    /// none.
    pub data_messages: &'a [DataMessage],
}

/// A non-radial message in a Level II source's data records (after the
/// metadata record), such as an RDA status (Message 2) update sent in the
/// middle of a volume.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataMessage {
    /// Radials before it in the source's records, the metadata record's
    /// included.
    pub after_radials: usize,
    /// Its fixed 2432-byte frames as the source stored them: the 12-byte
    /// CTM header, the message header and the body, one frame per segment.
    pub frames: Vec<u8>,
}

/// The non-radial messages of Level II `bytes`' data records, in order, each
/// with the number of radials before it (see [`SourceMetadata`]). Messages
/// of the metadata record are not included; its radials, which files from
/// before 2005 have there, are counted.
pub fn data_messages(bytes: &[u8]) -> Result<Vec<DataMessage>, crate::NexradError> {
    let metadata_len = crate::messages::metadata_record(bytes)?.len();
    // The records decompressed in parallel as the volume decoder does it;
    // inputs it does not take (no volume header) record by record.
    let normalized = crate::normalize_archive_bytes(bytes).map(|(normalized, _)| normalized);
    let records: std::borrow::Cow<'_, [u8]> = match normalized {
        Ok(mut normalized) => {
            normalized.drain(..crate::messages::volume_header_len(&normalized));
            std::borrow::Cow::Owned(normalized)
        }
        Err(_) => crate::messages::record_bytes(bytes)?,
    };
    let mut radials = 0usize;
    let mut messages = Vec::new();
    for item in crate::messages::RawMessages::new(&records) {
        let Ok(message) = item else { continue };
        let message_type = message.header.message_type;
        if matches!(message_type, 1 | 31) {
            radials += 1;
            continue;
        }
        let Some(frame_start) = message.offset.checked_sub(encode::CTM_LEN) else {
            continue;
        };
        if message_type == 0 || frame_start < metadata_len {
            continue;
        }
        let len = message.frames.saturating_mul(encode::FRAME_LEN);
        let end = frame_start.saturating_add(len).min(records.len());
        let mut frames = records[frame_start..end].to_vec();
        // The last frame of a file may end where its message does.
        frames.resize(len, 0);
        messages.push(DataMessage {
            after_radials: radials,
            frames,
        });
    }
    Ok(messages)
}

/// A Level II writer refusal. Nothing was produced when one is returned
/// (except for [`WriteError::Io`], which can happen mid-stream).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WriteError {
    /// The volume has no sweep with rays.
    #[error("the volume has no sweep with rays")]
    EmptyVolume,
    /// No ray of any sweep has data of a Message 31 moment: no field maps to
    /// one, or every mapped field was left out ([`WriteSummary::skipped_fields`]
    /// says why) or has no rows.
    #[error(
        "no ray has data of a Message 31 moment (REF, VEL, SW, ZDR, PHI, RHO, CFP); \
         see WriteOptions::field_map"
    )]
    NoMoments,
    /// A [`realtime::ChunkWriter`] was given a sweep with a moment its
    /// planned volume lacks, so no coding was fixed for it when the start
    /// chunk went out.
    #[error(
        "sweep {sweep}: {moment} (from {field}) is not in the planned volume, whose moments fix \
         the codings; plan with a volume that has it"
    )]
    UnplannedMoment {
        /// Sweep index in the pushed volume.
        sweep: usize,
        /// The moment.
        moment: Moment,
        /// The field that maps to it.
        field: String,
    },
    /// A field has values its moment's coding cannot hold, which would be
    /// clipped. The writer chooses each coding to hold every value of the
    /// moment, so this happens under a [`realtime::ChunkWriter`], whose
    /// codings the planned volume fixed before the data arrived (`planned`):
    /// plan with a volume whose values span the radar's (the same scan
    /// strategy and moments, or an earlier volume with a wider range).
    /// Nothing was written.
    #[error(
        "sweep {sweep}: {gates} gate(s) of {field} ({moment}) lie outside {low} to {high}, the \
         values the moment's {} coding holds; the writer does not clip values",
        if *planned { "planned" } else { "chosen" }
    )]
    ValueOutsideCoding {
        /// Sweep index in the volume (for a [`realtime::ChunkWriter`], in
        /// the pushed volume).
        sweep: usize,
        /// The moment.
        moment: Moment,
        /// The field that maps to it.
        field: String,
        /// Gates whose value lies outside the coding.
        gates: usize,
        /// Smallest value the coding holds.
        low: f32,
        /// Largest value the coding holds.
        high: f32,
        /// Whether the coding was fixed by a [`realtime::ChunkWriter`]'s
        /// planned volume.
        planned: bool,
    },
    /// A sweep is not a PPI-type scan.
    #[error("sweep {sweep}: {mode} sweeps cannot be written as Level II Message 31 cuts")]
    UnsupportedSweepMode {
        /// Sweep index in the volume.
        sweep: usize,
        /// The sweep mode.
        mode: String,
    },
    /// More elevation cuts than Level II numbers.
    #[error("{count} sweeps to write; Level II elevation numbers run from 1 to {max}")]
    TooManySweeps {
        /// Sweeps that would be written (with rays that have moment data).
        count: usize,
        /// The limit.
        max: usize,
    },
    /// The site identifier cannot be written.
    #[error("site identifier: {0}")]
    InvalidSiteId(String),
    /// A field's gate geometry cannot be written.
    #[error("sweep {sweep} field {field}: {reason}")]
    Geometry {
        /// Sweep index in the volume.
        sweep: usize,
        /// Field name.
        field: String,
        /// What does not fit.
        reason: String,
    },
    /// A ray's time or angles cannot be written.
    #[error("sweep {sweep} ray {ray}: {reason}")]
    Ray {
        /// Sweep index in the volume.
        sweep: usize,
        /// Ray index in the sweep.
        ray: usize,
        /// What does not fit.
        reason: String,
    },
    /// A sweep's per-ray arrays, or a written field's rows, do not match its
    /// ray count: the volume was built or changed without
    /// [`Sweep::seal`](recast_radar_core::model::Sweep::seal), which checks
    /// them.
    #[error("sweep {sweep}: {reason}; seal the sweep (Sweep::seal) before writing it")]
    Inconsistent {
        /// Sweep index in the volume.
        sweep: usize,
        /// What does not match.
        reason: String,
    },
    /// A radial would exceed the 65535-byte radial length of the Data Header
    /// Block.
    #[error(
        "sweep {sweep} ray {ray}: radial of {bytes} bytes exceeds the 65535-byte Message 31 limit"
    )]
    RadialTooLarge {
        /// Sweep index in the volume.
        sweep: usize,
        /// Ray index in the sweep.
        ray: usize,
        /// Radial length.
        bytes: usize,
    },
    /// The volume holds sweeps of more than one scan cycle, which one Level
    /// II file cannot hold: a cut collected again, a pause of minutes, or a
    /// second start of a Level II volume
    /// ([`scan_cycles`](recast_radar_core::model::scan_cycles)). Write each
    /// cycle on its own
    /// ([`split_scan_cycles`](recast_radar_core::model::split_scan_cycles)).
    #[error(
        "the volume holds more than one scan cycle ({begins}); a Level II file holds one volume \
         scan: write each cycle on its own (split_scan_cycles), or keep one cycle's sweeps"
    )]
    MixedScanCycles {
        /// Where the second cycle begins (sweep indices of the volume, or of
        /// the pushed sweeps for a [`realtime::ChunkWriter`]).
        begins: CycleBreak,
    },
    /// An option value is out of range.
    #[error("invalid option: {0}")]
    InvalidOption(String),
    /// The volume has no usable site location (`Volume::location`): every
    /// Message 31 VOL block and the RDA adaptation data carry the radar's
    /// latitude, longitude and height, and the writer does not invent them.
    /// Set `Volume::location` to the radar's position first (Message 1
    /// volumes, for example, carry none).
    #[error("the volume has no site {0}; set Volume::location to the radar's position")]
    MissingLocation(&'static str),
    /// The metadata record passed in [`SourceMetadata`] does not frame.
    #[error("source metadata record: {0}")]
    MetadataRecord(String),
    /// A data message passed in [`SourceMetadata::data_messages`] is not a
    /// non-radial message in whole fixed frames.
    #[error("source data message {index}: {reason}")]
    DataMessage {
        /// Index in [`SourceMetadata::data_messages`].
        index: usize,
        /// What is wrong with it.
        reason: String,
    },
    /// The output would exceed a resource limit.
    #[error("limit exceeded: {0}")]
    LimitExceeded(String),
    /// A compressor failed.
    #[error("compression: {0}")]
    Compression(String),
    /// The sink failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// How one field was written.
#[derive(Clone, Debug, PartialEq)]
pub struct MomentReport {
    /// Sweep index in the volume.
    pub sweep: usize,
    /// The Message 31 moment.
    pub moment: Moment,
    /// The source field.
    pub field: FieldName,
    /// Bits per gate (8 or 16).
    pub word_size: u8,
    /// Message 31 scale (`value = (code - offset) / scale`).
    pub scale: f32,
    /// Message 31 offset.
    pub offset: f32,
    /// `true` when every value decodes back to the source value (up to f32
    /// rounding of the coding): NEXRAD codes copied, or an exact grid.
    pub exact: bool,
    /// Largest difference between a source value and its decoded value.
    /// No value is ever clipped: a value the coding cannot hold is refused
    /// ([`WriteError::ValueOutsideCoding`]).
    pub max_abs_error: f32,
    /// Rays the source field did not provide (their moment block is left
    /// out of the radial).
    pub absent_rays: usize,
    /// Leading gates of every ray left out because their centre lies before
    /// the radar ([`WriteOptions::drop_negative_range_gates`]).
    pub dropped_gates: usize,
}

/// A field that was not written.
#[derive(Clone, Debug, PartialEq)]
pub struct SkippedField {
    /// Sweep index in the volume.
    pub sweep: usize,
    /// Field name.
    pub field: FieldName,
    /// Why.
    pub reason: String,
}

/// The rays of a sweep that was not written as its rays in storage order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WrittenRays {
    /// Sweep index in the volume.
    pub sweep: usize,
    /// The source ray of each written radial, in the order written. A sweep
    /// whose storage order is the order of collection turned round (its ray
    /// times, to the millisecond, run forward but for one step back, and
    /// its last ray is no later than its first: an ODIM sweep stored from
    /// north while the antenna started elsewhere) is written from its
    /// earliest ray; a Level II sweep, and any other, keeps its order. Rays
    /// on which no written moment has data are not listed.
    pub rays: Vec<usize>,
}

/// What a write produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WriteSummary {
    /// Bytes written.
    pub bytes: usize,
    /// Site identifier written.
    pub icao: String,
    /// Volume header time: the earliest written radial's (a Level II
    /// source's own header time when its metadata is carried over).
    pub volume_time: Option<DateTime<Utc>>,
    /// Elevation cuts written.
    pub sweeps: usize,
    /// The source sweep of each written cut, in cut order (elevation number
    /// 1 first): a foreign volume's in the order they were collected, a
    /// Level II source's as given.
    pub written_sweeps: Vec<usize>,
    /// Message 31 radials written.
    pub radials: usize,
    /// Records written (the metadata record included).
    pub records: usize,
    /// One entry per written field.
    pub moments: Vec<MomentReport>,
    /// Fields left out.
    pub skipped_fields: Vec<SkippedField>,
    /// Sweeps left out (source indices): sweeps without rays, and sweeps on
    /// whose rays no written moment has data (a note says which).
    pub skipped_sweeps: Vec<usize>,
    /// Sweeps whose radials are not their rays in storage order: rays
    /// written in time order, or rays without data left out.
    pub written_rays: Vec<WrittenRays>,
    /// Largest range error of any written field, metres at its last gate.
    pub max_range_error_m: f64,
    /// Other remarks (for example an ignored metadata record).
    pub notes: Vec<String>,
}

/// Write `volume` as Archive II bytes.
pub fn write_volume(volume: &Volume, options: &WriteOptions) -> Result<Vec<u8>, WriteError> {
    write_volume_with_source(volume, SourceMetadata::default(), options).map(|(bytes, _)| bytes)
}

/// Write `volume` as Archive II bytes into `out`, returning what was written.
///
/// The file is streamed: everything a refusal depends on is decided before
/// the first byte (a refused volume writes nothing), then the volume header,
/// the metadata record and the radial records go to `out` a batch at a time
/// (one record per rayon thread, compressed in parallel), so the whole file
/// is never held in memory: besides the volume itself, the writer holds its
/// plan (codings and per-ray values) and one batch of records, compressed
/// and not. A sink, compressor or allocation failure after the first byte
/// ([`WriteError::Io`], [`WriteError::Compression`],
/// [`WriteError::LimitExceeded`]) leaves a partial file in `out`. The bytes
/// are those [`write_volume`] returns.
pub fn write_volume_to<W: Write>(
    volume: &Volume,
    options: &WriteOptions,
    out: &mut W,
) -> Result<WriteSummary, WriteError> {
    write_volume_with_source_to(volume, SourceMetadata::default(), options, out)
}

/// [`write_volume_to`] reusing Level II context from the volume's source
/// ([`SourceMetadata`]).
pub fn write_volume_with_source_to<W: Write>(
    volume: &Volume,
    source: SourceMetadata<'_>,
    options: &WriteOptions,
    out: &mut W,
) -> Result<WriteSummary, WriteError> {
    let plan = plan::plan(volume, source, options)?;
    let mut counted = CountingWriter {
        inner: out,
        bytes: 0,
    };
    let records = if options.gzip {
        let mut gzip = compress::gzip_writer(&mut counted);
        let records = stream_archive(&plan, options.compression, &mut gzip)?;
        gzip.finish()?;
        records
    } else {
        stream_archive(&plan, options.compression, &mut counted)?
    };
    let bytes = counted.bytes;
    let mut summary = plan.summary;
    summary.records = records;
    summary.bytes = bytes;
    Ok(summary)
}

/// Write `volume` as Archive II bytes, reusing Level II context from its
/// source ([`SourceMetadata`]).
pub fn write_volume_with_source(
    volume: &Volume,
    source: SourceMetadata<'_>,
    options: &WriteOptions,
) -> Result<(Vec<u8>, WriteSummary), WriteError> {
    let mut buffer = BufferSink(Vec::new());
    let summary = write_volume_with_source_to(volume, source, options, &mut buffer).map_err(
        |err| match err {
            WriteError::Io(err) if err.kind() == std::io::ErrorKind::OutOfMemory => {
                WriteError::LimitExceeded(format!("output buffer: {err}"))
            }
            other => other,
        },
    )?;
    Ok((buffer.0, summary))
}

/// The volume header, the metadata record and the radial records of `plan`
/// written to `out` as records of `compression`, a batch at a time; returns
/// the number of records written (the metadata record included).
fn stream_archive<W: Write>(
    plan: &plan::Plan<'_>,
    compression: Compression,
    out: &mut W,
) -> Result<usize, WriteError> {
    out.write_all(&encode::volume_header(plan))?;
    let mut records = RecordStream {
        compression,
        compressor: compress::LdmCompressor::new(),
        out,
        pending: Vec::new(),
        batch: rayon::current_num_threads().max(1),
        written: 0,
    };
    records.push(encode::metadata_record_bytes(plan))?;
    let context = encode::RadialContext {
        opens_volume: true,
        last_cut: plan.sweeps.last().map_or(0, |sweep| sweep.elevation_number),
        ends_volume: true,
    };
    let mut sequence = encode::FIRST_RADIAL_SEQUENCE;
    encode::for_each_radial_record(
        plan,
        &context,
        &mut sequence,
        plan.radials_per_record,
        |record| records.push(record.bytes),
    )?;
    records.finish()
}

/// Records on their way to the sink: held until a batch is complete, then
/// compressed together and written. The last record of the file is held
/// until [`RecordStream::finish`], because its LDM control word is negated.
struct RecordStream<'w, W: Write> {
    compression: Compression,
    /// The encoders of every batch of this file.
    compressor: compress::LdmCompressor,
    out: &'w mut W,
    pending: Vec<Vec<u8>>,
    batch: usize,
    written: usize,
}

impl<W: Write> RecordStream<'_, W> {
    fn push(&mut self, record: Vec<u8>) -> Result<(), WriteError> {
        if self.pending.len() >= self.batch {
            self.flush(false)?;
        }
        self.pending
            .try_reserve(1)
            .map_err(|err| WriteError::LimitExceeded(format!("record batch: {err}")))?;
        self.pending.push(record);
        Ok(())
    }

    /// Write the pending records; `ends_file` when the last of them is the
    /// file's last record.
    fn flush(&mut self, ends_file: bool) -> Result<(), WriteError> {
        let records = std::mem::take(&mut self.pending);
        let stored = match self.compression {
            Compression::None => records,
            Compression::Bzip2LdmRecords => self.compressor.batch(&records, ends_file)?,
        };
        for record in &stored {
            self.out.write_all(record)?;
        }
        self.written += stored.len();
        Ok(())
    }

    fn finish(mut self) -> Result<usize, WriteError> {
        self.flush(true)?;
        Ok(self.written)
    }
}

/// A sink that counts the bytes written through it.
struct CountingWriter<'w, W: Write> {
    inner: &'w mut W,
    bytes: usize,
}

impl<W: Write> Write for CountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.bytes += written;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// An in-memory sink whose growth failing is an error
/// ([`std::io::ErrorKind::OutOfMemory`]), not an abort.
struct BufferSink(Vec<u8>);

impl Write for BufferSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .try_reserve(buf.len())
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::OutOfMemory, err))?;
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Decode Level II `bytes` (any compression) with their metadata and write
/// them again with `options`: a re-encoding that keeps the metadata record
/// (made to agree with the options, see [`SourceMetadata`]), the non-radial
/// messages of the data records at their places among the radials, every
/// radial's constant blocks and header items (its radar identifier and
/// azimuth resolution code included), the volume header time and every gate
/// code. What changes: the records' grouping (records of
/// `options.radials_per_record` radials, by `options.record_layout`) and
/// compression, the message sequence numbers of the radials, the radial
/// statuses that place a radial in the volume, and what the options set
/// (the header's volume number; with `options.icao`, the site in the header,
/// in Message 18 and in every radial, whose own identifiers are otherwise
/// kept, blank ones included); see `docs/level2/writer.md`.
pub fn rewrite_level2(
    bytes: &[u8],
    options: &WriteOptions,
) -> Result<(Vec<u8>, WriteSummary), RewriteError> {
    let decoded = crate::read_volume_with_metadata(bytes)?;
    let record = crate::messages::metadata_record(bytes)?;
    let messages = data_messages(bytes)?;
    let source = SourceMetadata {
        metadata: Some(&decoded.metadata),
        metadata_record: Some(record.as_ref()),
        data_messages: &messages,
    };
    Ok(write_volume_with_source(&decoded.volume, source, options)?)
}

/// Error of [`rewrite_level2`]: the decode or the write failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RewriteError {
    /// Decoding the input failed.
    #[error("decode: {0}")]
    Decode(#[from] crate::NexradError),
    /// Writing failed.
    #[error("write: {0}")]
    Write(#[from] WriteError),
}

/// The encoded archive for the real-time chunker: the header and each
/// record's bytes as stored.
pub(crate) struct Archive {
    pub header: [u8; encode::VOLUME_HEADER_LEN],
    /// Each record as stored in the file (control word and bzip2 stream,
    /// or the uncompressed frames).
    pub records: Vec<Vec<u8>>,
    pub summary: WriteSummary,
    pub volume_number: u16,
}

/// Plan, encode and compress `volume`, every record at once.
pub(crate) fn build_archive(
    volume: &Volume,
    source: SourceMetadata<'_>,
    options: &WriteOptions,
) -> Result<Archive, WriteError> {
    let plan = plan::plan(volume, source, options)?;
    let encoded = encode::encode(&plan)?;
    let stored = match options.compression {
        Compression::None => encoded.records,
        Compression::Bzip2LdmRecords => {
            compress::LdmCompressor::new().batch(&encoded.records, true)?
        }
    };
    let mut summary = plan.summary;
    summary.records = stored.len();
    Ok(Archive {
        header: encoded.header,
        records: stored,
        summary,
        volume_number: plan.volume_number,
    })
}
