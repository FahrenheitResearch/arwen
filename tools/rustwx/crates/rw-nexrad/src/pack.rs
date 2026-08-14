//! Strict Level-II framing validation and the flat sweep pack.
//!
//! `wx_radar::level2::Level2File::parse` is forgiving by design — it is the
//! renderer's parser, and a renderer that draws 95% of a damaged volume
//! beats one that draws nothing.  An observation front door has the
//! opposite duty, so it reaches for `parse_strict` instead, and every
//! volume is framed-checked *before* it reaches the parser at all: a
//! truncated Archive-II block table, a gzip-wrapped legacy volume, or a
//! file that is not Archive-II at all is refused with the byte offsets that
//! prove it, never silently half-decoded.
//!
//! The two checks catch different things and neither subsumes the other.
//! Framing walks the outer LDM block table, whose lengths can be perfectly
//! valid while an inner Message-31 pointer aims into the next radial;
//! `parse_strict` bounds every block read to the radial that declared it,
//! but only sees bytes the framing already accepted.
//!
//! The pack itself is deliberately dull: a 64-byte little-endian header, a
//! JSON metadata block, and one contiguous payload of `<f4` arrays.  It is
//! the `.rwg` layout from `rw-store` (magic / version / meta_len / payload)
//! carried over so a Python consumer needs `json` and `numpy.frombuffer`
//! and nothing else.

use std::borrow::Cow;
use std::error::Error;
use std::io::Read;
use std::path::Path;

use rw_store::atomic::atomic_write_bytes;
use serde::{Deserialize, Serialize};
use wx_radar::level2::Level2File;
use wx_radar::products::RadarProduct;

use crate::s3::{boxed_error, hex_sha256};

/// Pack file magic — eight bytes, versioned by the `u32` that follows.
pub const PACK_MAGIC: &[u8; 8] = b"GPWMRDR1";
pub const PACK_VERSION: u32 = 1;
pub const PACK_HEADER_BYTES: usize = 64;
/// The schema the pack metadata declares, and the contract the Python
/// sweep reader checks before it touches a byte of payload.
pub const SWEEPS_SCHEMA: &str = "gpuwm-obs.radar-sweeps.v1";

/// The schema a pack declares once it also carries censor planes.
///
/// A separate string rather than a flag inside `v1`, because a `v1` reader
/// that silently ignored the extra arrays would be reading a pack whose
/// moment planes it can no longer fully account for: in `v1` a `NaN` gate is
/// "nothing to say about this gate", and in `v2` the pack is asserting it
/// knows exactly which of three things happened there.  The readers refuse
/// what they do not know (`decode_pack` here, `gpuwm.obs.sweeps` on the
/// Python side), so the version bump is what makes the extension safe rather
/// than merely additive.
///
/// A `v2` pack is otherwise byte-for-byte a `v1` pack with more arrays: same
/// magic, same 64-byte header, same `PACK_VERSION`, same moment planes.  The
/// container did not change, so the container's version did not either.
pub const SWEEPS_SCHEMA_CENSOR: &str = "gpuwm-obs.radar-sweeps.v2";

/// The same container again, written by `rw_odim` from an ODIM_H5 polar
/// volume.
///
/// A third schema string rather than `v2`, for the reason `v2` is not `v1`:
/// the censor vocabulary is genuinely wider.  ODIM distinguishes two states
/// NEXRAD has no word for — `NODATA` (4), the radar did not measure this
/// gate, and `SENTINEL_AMBIGUOUS` (5), the file declares its two sentinels
/// as the same raw value so a gate holding it may be either.  A `v2` reader
/// handed either code would be reading a plane it cannot fully account for,
/// and the schema string is what stops it.
///
/// The codes that mean the same thing are the same number, which is what
/// lets one lookup table serve both radar families.
pub const SWEEPS_SCHEMA_ODIM: &str = "gpuwm-obs.radar-sweeps.v3";

/// `<u1` census planes, one code per gate.  Mirrors
/// [`wx_radar::level2::censor`], plus the one code only this layer can mint.
pub mod censor_plane {
    pub const DTYPE: &str = "|u1";
    /// A radial that carried no such moment at all.  The pack stores a sweep
    /// as a rectangle, and `build_pack` fills the rows a split cut left empty
    /// with `NaN`; this is the code that says so, as against a gate the radar
    /// actually looked at and found empty.
    pub const NOT_COLLECTED: u8 = wx_radar::level2::censor::NOT_COLLECTED;
}

fn is_false(value: &bool) -> bool {
    !*value
}

const VOLUME_HEADER_SIZE: usize = 24;
/// Offset past which a message header is fully readable: 12 bytes of CTM
/// then the 16-byte message header, of which the size word and type byte
/// are the first four.
const MESSAGE_HEADER_END: usize = 16;
/// The legacy record every non-Message-31 message occupies.
const LEGACY_RECORD_BYTES: usize = 2432;

/// How the message stream was carried inside the volume file.
///
/// Both spellings are Archive-II and the magic does not tell them apart.
pub mod layout {
    /// A four-byte block length then a bzip2 stream, repeated: every plain
    /// `_V06` key on the open-data buckets, roughly 2017 onward.
    pub const LDM_BZIP2: &str = "ldm-bzip2";
    /// Messages straight after the 24-byte volume header, no block table:
    /// what the pre-2016 `.gz` keys hold once gunzipped.
    pub const UNCOMPRESSED_MESSAGES: &str = "uncompressed-messages";
}

/// What [`read_volume`] proved about a volume.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Framing {
    /// Archive-II version token from the first nine bytes, e.g. `AR2V0006`.
    pub magic: String,
    /// One of [`layout::LDM_BZIP2`] or [`layout::UNCOMPRESSED_MESSAGES`].
    pub layout: String,
    /// True when the file on disk was gzip-wrapped and had to be expanded
    /// before any of the above could be read.
    pub gzip_wrapped: bool,
    /// Number of LDM blocks between the volume header and end of file.
    /// Zero for [`layout::UNCOMPRESSED_MESSAGES`], which has no block table.
    pub block_count: usize,
    /// How many of those blocks carry a `BZh` bzip2 stream.
    pub bzip2_block_count: usize,
    /// Messages the framing walk tiled, for the unframed layout.  Zero for
    /// `ldm-bzip2`, whose messages are inside the compressed blocks and are
    /// counted by the decoder rather than by the framing check.
    pub message_count: usize,
    /// Bytes of the message stream that was validated, after any gunzip.
    pub bytes: usize,
    /// Bytes of the file exactly as it arrived.  Equal to `bytes` unless
    /// the file was gzip-wrapped.  This is the number the S3 listing states
    /// and the number the download's sha256 is taken over.
    pub source_bytes: usize,
}

/// Largest message stream this front door will expand a `.gz` volume into.
///
/// A 2013 volume is about 8 MB gzipped and 45 MB expanded, so 512 MB is an
/// order of magnitude of headroom over anything the archive holds.  It is
/// not a tuning parameter, it is the bound that stops a malformed or
/// hostile `.gz` from being decompressed until the process dies.
const MAX_EXPANDED_VOLUME_BYTES: u64 = 512 * 1024 * 1024;

/// Expand a gzip-wrapped volume, or hand back the bytes unchanged.
///
/// The archive stores roughly 2011-2016 as `..._V06.gz` and everything
/// after plain, so a front door that reads only one of the two spellings
/// cannot see half the record. The wrapper is a storage detail: what comes
/// out is an ordinary Archive-II volume and every strict check downstream
/// runs over it unchanged.
///
/// The bytes as they arrived stay the volume's identity -- the S3 listing's
/// size and the download's sha256 are both taken over those, not over the
/// expansion -- so only interpretation happens here.
pub fn gunzip_if_wrapped(raw: &[u8]) -> Result<(Cow<'_, [u8]>, bool), Box<dyn Error>> {
    if !raw.starts_with(&[0x1f, 0x8b]) {
        return Ok((Cow::Borrowed(raw), false));
    }
    let mut expanded = Vec::new();
    let mut decoder = flate2::read::GzDecoder::new(raw).take(MAX_EXPANDED_VOLUME_BYTES + 1);
    decoder.read_to_end(&mut expanded).map_err(|err| {
        boxed_error(format!(
            "gzip-wrapped Level-II volume will not expand: {err}"
        ))
    })?;
    if expanded.len() as u64 > MAX_EXPANDED_VOLUME_BYTES {
        return Err(boxed_error(format!(
            "gzip-wrapped Level-II volume expands past the {MAX_EXPANDED_VOLUME_BYTES}-byte \
             ceiling; refusing rather than decompressing without bound"
        )));
    }
    // A truncated member does not necessarily fail to decode: a deflate
    // stream can end on a block boundary with the 8-byte trailer missing,
    // and the reader then returns the bytes it got and no error. Measured
    // on this flate2 build, so it is checked rather than assumed. The
    // trailer's ISIZE is the member's own statement of how many bytes it
    // expands to, and it is the last thing a truncation removes.
    if raw.len() < 18 {
        return Err(boxed_error(format!(
            "gzip-wrapped Level-II volume is {} bytes, too few to hold a gzip member's header \
             and trailer",
            raw.len()
        )));
    }
    let tail = &raw[raw.len() - 4..];
    let declared = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
    if declared != (expanded.len() as u64 % (1u64 << 32)) as u32 {
        return Err(boxed_error(format!(
            "gzip-wrapped Level-II volume is incomplete: its trailer declares {declared} \
             expanded bytes and {} came out. A member cut short can still decode to a \
             prefix without erroring, and a prefix of a volume is not a short volume -- it \
             is the first N radials with no way to tell how many are missing",
            expanded.len()
        )));
    }
    Ok((Cow::Owned(expanded), true))
}

/// Read a volume file: expand it if gzipped, then prove its framing.
///
/// Refuses anything that is not a whole, well-framed Archive-II volume in
/// one of the two shapes the archive actually holds.  The check that
/// matters is the same in both: the structure must consume the file
/// exactly -- the LDM block table for `ldm-bzip2`, the message walk for
/// `uncompressed-messages`.  A truncated download breaks that walk and
/// leaves trailing bytes unaccounted for, which is precisely the failure
/// the forgiving parser would paper over by returning the sweeps it
/// managed to read.
///
/// Returns the message stream to parse and the framing record to publish.
/// `source` must be the file exactly as it arrived, because that is what
/// the identity fields in the record are taken over.
pub fn read_volume(source: &[u8]) -> Result<(Cow<'_, [u8]>, Framing), Box<dyn Error>> {
    let (stream, gzip_wrapped) = gunzip_if_wrapped(source)?;
    let framing = validate_stream_framing(&stream, gzip_wrapped, source.len())?;
    Ok((stream, framing))
}

fn validate_stream_framing(
    raw: &[u8],
    gzip_wrapped: bool,
    source_bytes: usize,
) -> Result<Framing, Box<dyn Error>> {
    if raw.len() < VOLUME_HEADER_SIZE {
        return Err(boxed_error(format!(
            "not a Level-II volume: {} bytes, the 24-byte volume header alone needs more",
            raw.len()
        )));
    }
    if raw.starts_with(&[0x1f, 0x8b]) {
        return Err(boxed_error(
            "gzip-wrapped Level-II volume reached the framing check still compressed; \
             gunzip_if_wrapped should have expanded it first"
                .to_string(),
        ));
    }
    let magic = String::from_utf8_lossy(&raw[..8]).to_string();
    if !(magic.starts_with("AR2V") || magic.starts_with("ARCH")) {
        return Err(boxed_error(format!(
            "not a Level-II volume: leading bytes {magic:?}, expected AR2V.... or ARCH...."
        )));
    }

    // Which of the two real shapes is this?  The first block's payload
    // decides: an LDM block table always opens with the bzip2 metadata
    // block, and in the unframed shape those bytes are the first message's
    // CTM header.  `wx_radar::Level2File` makes the same decision the same
    // way, and the two must agree or the framing record describes a volume
    // the decoder did not read.
    if !is_ldm_framed(raw) {
        return validate_uncompressed_messages(raw, magic, gzip_wrapped, source_bytes);
    }

    let mut pos = VOLUME_HEADER_SIZE;
    let mut block_count = 0usize;
    let mut bzip2_block_count = 0usize;
    while pos < raw.len() {
        if pos + 4 > raw.len() {
            return Err(boxed_error(format!(
                "truncated Level-II volume: {} trailing bytes at offset {pos} cannot hold a \
                 4-byte LDM block length",
                raw.len() - pos
            )));
        }
        let declared = i32::from_be_bytes([raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]]);
        pos += 4;
        let size = declared.unsigned_abs() as usize;
        if size == 0 {
            return Err(boxed_error(format!(
                "corrupt Level-II volume: LDM block {block_count} at offset {} declares zero bytes",
                pos - 4
            )));
        }
        if pos + size > raw.len() {
            return Err(boxed_error(format!(
                "truncated Level-II volume: LDM block {block_count} at offset {} declares {size} \
                 bytes but only {} remain",
                pos - 4,
                raw.len() - pos
            )));
        }
        if size >= 3 && &raw[pos..pos + 3] == b"BZh" {
            bzip2_block_count += 1;
        }
        pos += size;
        block_count += 1;
    }
    // Two refusals used to sit here -- "no LDM blocks follow" and "none of
    // them a bzip2 stream" -- and both are now unreachable: `is_ldm_framed`
    // only sends a volume down this path when the first block exists, fits,
    // and opens `BZh`, which is exactly the condition each of them tested.
    // A volume that fails either is classified as an uncompressed message
    // stream instead and refused there, by what its bytes actually look
    // like. They are deleted rather than kept as gates that cannot fire.
    debug_assert!(block_count >= 1 && bzip2_block_count >= 1);
    Ok(Framing {
        magic,
        layout: layout::LDM_BZIP2.to_string(),
        gzip_wrapped,
        block_count,
        bzip2_block_count,
        message_count: 0,
        bytes: raw.len(),
        source_bytes,
    })
}

/// True when the bytes after the volume header are an LDM block table.
///
/// The same test `wx_radar::Level2File::is_ldm_framed` applies, kept here
/// rather than imported so the framing record and the decoder each state
/// the rule they used; `the_two_crates_classify_the_same_bytes_alike`
/// pins them together.
fn is_ldm_framed(raw: &[u8]) -> bool {
    if raw.len() < VOLUME_HEADER_SIZE + 4 + 3 {
        return false;
    }
    let declared = i32::from_be_bytes([
        raw[VOLUME_HEADER_SIZE],
        raw[VOLUME_HEADER_SIZE + 1],
        raw[VOLUME_HEADER_SIZE + 2],
        raw[VOLUME_HEADER_SIZE + 3],
    ]);
    let size = declared.unsigned_abs() as usize;
    let start = VOLUME_HEADER_SIZE + 4;
    size >= 3 && start + size <= raw.len() && &raw[start..start + 3] == b"BZh"
}

/// Prove the pre-2016 shape: messages tile the file exactly, from the end
/// of the volume header to the last byte.
///
/// This is the same completeness argument the LDM walk makes, against the
/// only structure this layout has. A truncated download leaves the walk
/// short of the end or steps past it, and either way the remainder is not
/// zero. The step is the one `wx_radar`'s strict `read_message` takes --
/// the declared size for a Message-31, a 2432-byte legacy record for
/// anything else -- so the framing check and the decoder walk the same
/// bytes in the same strides rather than two plausible ways.
fn validate_uncompressed_messages(
    raw: &[u8],
    magic: String,
    gzip_wrapped: bool,
    source_bytes: usize,
) -> Result<Framing, Box<dyn Error>> {
    let mut pos = VOLUME_HEADER_SIZE;
    let mut message_count = 0usize;
    let mut radial_count = 0usize;
    while pos < raw.len() {
        if pos + MESSAGE_HEADER_END > raw.len() {
            return Err(boxed_error(format!(
                "truncated Level-II volume: {} trailing bytes at offset {pos} cannot hold a \
                 message header, and this volume has no LDM block table to check instead \
                 (the first block after the header does not carry a bzip2 stream, so it was \
                 read as an uncompressed message stream)",
                raw.len() - pos
            )));
        }
        let declared = u16::from_be_bytes([raw[pos + 12], raw[pos + 13]]) as usize;
        let message_type = raw[pos + 15];
        let step = if message_type == 31 {
            radial_count += 1;
            declared * 2 + 12
        } else {
            LEGACY_RECORD_BYTES
        };
        if step == 0 {
            return Err(boxed_error(format!(
                "corrupt Level-II volume: the message at offset {pos} declares zero bytes"
            )));
        }
        if pos + step > raw.len() {
            return Err(boxed_error(format!(
                "truncated Level-II volume: the message at offset {pos} declares {step} bytes \
                 but only {} remain",
                raw.len() - pos
            )));
        }
        pos += step;
        message_count += 1;
    }
    if radial_count == 0 {
        return Err(boxed_error(format!(
            "corrupt Level-II volume: {message_count} messages tile the file and none is a \
             Message-31 radial; there is no moment data here to observe"
        )));
    }
    Ok(Framing {
        magic,
        layout: layout::UNCOMPRESSED_MESSAGES.to_string(),
        gzip_wrapped,
        block_count: 0,
        bzip2_block_count: 0,
        message_count,
        bytes: raw.len(),
        source_bytes,
    })
}

/// Refuse a volume that framed cleanly but decoded to nothing usable.
pub fn validate_decoded(file: &Level2File) -> Result<(), Box<dyn Error>> {
    if file.station_id.len() != 4 || !file.station_id.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(boxed_error(format!(
            "corrupt Level-II volume: station id {:?} is not a four-character site",
            file.station_id
        )));
    }
    if file.volume_date == 0 {
        return Err(boxed_error(
            "corrupt Level-II volume: volume date is zero".to_string(),
        ));
    }
    if file.sweeps.is_empty() {
        return Err(boxed_error(
            "corrupt Level-II volume: decoded to zero sweeps".to_string(),
        ));
    }
    let gates: usize = file
        .sweeps
        .iter()
        .flat_map(|sweep| sweep.radials.iter())
        .flat_map(|radial| radial.moments.iter())
        .map(|moment| moment.data.len())
        .sum();
    if gates == 0 {
        return Err(boxed_error(format!(
            "corrupt Level-II volume: {} sweeps decoded but not one range gate",
            file.sweeps.len()
        )));
    }
    Ok(())
}

/// One `<f4` array inside the payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayEntry {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentEntry {
    pub product: String,
    pub unit: String,
    pub gate_count: usize,
    pub first_gate_range_m: f64,
    pub gate_size_m: f64,
    /// Key into `arrays`; shape `[radial_count, gate_count]`.
    pub array: String,
    /// Key into `arrays` for the `|u1` censor plane, same shape as `array`.
    /// Present only in a [`SWEEPS_SCHEMA_CENSOR`] pack.
    ///
    /// `skip_serializing_if` is load-bearing rather than tidy: it is what
    /// keeps a default-mode pack's metadata JSON byte-identical to the one
    /// this build's predecessor wrote, and therefore keeps every committed
    /// pack digest reproducible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub censor_array: Option<String>,
    /// The producer's own name for this moment, when it differs from
    /// `product`.
    ///
    /// `product` is the pack's *consumer* vocabulary and there is exactly one
    /// of it: `REF`, `VEL`, and the superob layer keys on those two tokens.
    /// A producer whose archive spells them differently -- ODIM's `DBZH` and
    /// `VRADH` -- must translate, or every gate it writes is silently skipped
    /// by a consumer that still reports the sweep as used.  This field is how
    /// the translation stays lossless: the canonical token drives the
    /// pipeline and the original spelling stays readable, so a reader can
    /// still tell a `DBZH` from a `TH`.
    ///
    /// `Option` with `skip_serializing_if`, so a NEXRAD pack -- whose
    /// producer already speaks the consumer vocabulary -- is byte-identical
    /// to what it was before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_quantity: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepEntry {
    pub sweep_index: u16,
    pub elevation_number: u8,
    pub elevation_angle_deg: f64,
    /// The smallest Nyquist velocity any radial in the cut reported, not
    /// the first one seen: a sweep-wide scalar must not license a gate its
    /// own radial would have rejected.
    ///
    /// Since the pack began carrying [`Self::nyquist_by_radial_array`] this
    /// is *defined* as that array's minimum over its usable entries rather
    /// than computed beside it.  Two fields claiming the same thing are a
    /// defect the moment they can disagree, so there is one source and one
    /// reduction, and [`decode_pack`] refuses a pack where they part.
    pub nyquist_velocity_ms: Option<f64>,
    /// True when the cut's radials did not all report that same value.
    ///
    /// Derivable from [`Self::nyquist_by_radial_array`] once that is
    /// present, and kept anyway: it is what a pack written before the array
    /// existed said, and dropping it would silently downgrade every reader
    /// that already asks the question.
    #[serde(default)]
    pub nyquist_radials_disagree: bool,
    pub start_status: u8,
    pub end_status: u8,
    pub cut_sector: u8,
    /// `true` when the cut both started and ended on a status marker.
    pub complete: bool,
    pub radial_count: usize,
    /// Key into `arrays`; shape `[radial_count]`.
    pub azimuth_array: String,
    /// Key into `arrays`; shape `[radial_count]`.
    pub elevation_array: String,
    /// Key into `arrays` for the Nyquist velocity each radial reported;
    /// `<f4`, shape `[radial_count]`, in radial order, `NaN` where that
    /// radial's Message-31 RAD block carried none.
    ///
    /// The fold arithmetic a dealiaser does is per radial, and a cut whose
    /// radials sit on different Nyquist lattices cannot be described by one
    /// number — the scalar above is a floor, which is safe but is not the
    /// lattice.  This is the array that carries the originals.
    ///
    /// Absent when not one radial in the cut reported a usable value: the
    /// scalar is `None` there too, and an array of nothing but `NaN` would
    /// assert that originals were kept when there were none to keep.  Its
    /// absence and an all-`NaN` array are different statements and are kept
    /// different.  `skip_serializing_if` is load-bearing rather than tidy:
    /// it is what keeps such a pack's metadata JSON byte-identical to the
    /// one this build's predecessor wrote.
    ///
    /// The container did not change to carry it — same magic, same header,
    /// same [`PACK_VERSION`], same schema string.  A reader that has never
    /// heard of the key reads exactly the pack it always read, which is why
    /// this is additive rather than a schema bump: unlike the censor planes,
    /// it makes no new claim about any byte that was already there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nyquist_by_radial_array: Option<String>,
    /// What granularity the source file recorded the Nyquist interval at:
    /// `"radial"` or `"sweep"`.
    ///
    /// The two radar families genuinely differ here and the pack has to say
    /// which, because the dealiaser refuses by name when the array it needs
    /// is absent and that refusal has to be able to distinguish "this file
    /// never had per-radial originals" from "they were dropped".  A NEXRAD
    /// Message-31 RAD block carries one per radial; ODIM declares a single
    /// `/datasetN/how/NI` for the whole cut, so an ODIM pack states `"sweep"`
    /// and omits [`Self::nyquist_by_radial_array`] rather than broadcasting
    /// one number across the radials, which would assert that originals were
    /// kept when there was only ever one original.
    ///
    /// `None` on a pack written before the key existed; NEXRAD leaves it
    /// `None` so those packs stay byte-identical, and the presence of the
    /// per-radial array already says `"radial"` there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nyquist_granularity: Option<String>,
    pub moments: Vec<MomentEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteEntry {
    pub id: String,
    pub name: String,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_m: f64,
    /// Where the coordinates came from: the vendored NEXRAD table or the
    /// caller's `--site-latlon` override.  A superob must never guess.
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeParams {
    pub moments: Vec<String>,
    pub max_range_km: f64,
    pub max_elevation_deg: f64,
    /// Whether `--censor-flags` was asked for.  Omitted when false so a
    /// default-mode pack's metadata is unchanged from before the flag
    /// existed.
    #[serde(default, skip_serializing_if = "is_false")]
    pub censor_flags: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeEntry {
    pub file: String,
    pub bytes: usize,
    pub sha256: String,
    pub station_id: String,
    pub valid_time: String,
    pub volume_date: u16,
    pub volume_time_ms: u32,
    /// Archive-II framing, for a volume that had any.
    ///
    /// `Option` rather than required because ODIM_H5 is an HDF5 container
    /// with no LDM block table and no `AR2V` token: there is no honest
    /// `Framing` to write for one, and inventing a placeholder would put a
    /// claim about Archive-II layout into a file that is not Archive-II.
    /// `skip_serializing_if` keeps every NEXRAD pack byte-identical, and the
    /// Python reader already spelled this field `volume.get("framing") or {}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub framing: Option<Framing>,
    /// The member files, for a volume that was assembled out of several.
    ///
    /// `Option` for the same reason `framing` is: a NEXRAD volume is one
    /// file, so there is nothing honest to write, and `skip_serializing_if`
    /// keeps every NEXRAD pack byte-identical.  Some national ODIM feeds
    /// publish one file per elevation and quantity, and for those the
    /// `sha256` above is a manifest digest rather than a file digest -- this
    /// block is what says so and what lets a reader name the actual bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assembled: Option<AssembledEntry>,
}

/// The files an assembled volume was built from, and how they folded.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembledEntry {
    pub files: usize,
    /// Files whose cut identity matched an earlier member's, so their moments
    /// joined that sweep instead of starting a new one.  `files - merged` is
    /// the sweep count, and the ratio is a measurement of whether the feed
    /// records its moments on one antenna pass or on several.
    pub sweeps_merged: usize,
    /// SHA-256 over `"<sha256>  <name>\n"` lines sorted by name.  Identifies
    /// the member set; it is not any file's digest.
    pub manifest_sha256: String,
    pub members: Vec<AssembledMember>,
}

/// One member file of an assembled volume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembledMember {
    pub name: String,
    pub bytes: usize,
    pub sha256: String,
}

/// The pack metadata block — schema, provenance, geometry, array index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackMeta {
    pub schema: String,
    pub status: String,
    pub site: SiteEntry,
    pub volume: VolumeEntry,
    pub params: DecodeParams,
    pub sweeps: Vec<SweepEntry>,
    pub arrays: std::collections::BTreeMap<String, ArrayEntry>,
    pub payload_bytes: usize,
    pub content_sha256: String,
    /// Sweeps and moments dropped by `--moments` / `--max-elevation-deg`,
    /// reported so a thin pack is never mistaken for a thin volume.
    pub dropped_sweeps: usize,
    pub dropped_moments: usize,
    /// Gates trimmed off the far end by `--max-range-km`.
    pub trimmed_gates: usize,
}

/// Accumulates `<f4` arrays into one contiguous payload.
pub struct PayloadBuilder {
    payload: Vec<u8>,
    arrays: std::collections::BTreeMap<String, ArrayEntry>,
    next: usize,
}

impl Default for PayloadBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PayloadBuilder {
    pub fn new() -> Self {
        Self {
            payload: Vec::new(),
            arrays: std::collections::BTreeMap::new(),
            next: 0,
        }
    }

    /// Append one f32 array, returning its key.
    pub fn push_f32(&mut self, values: &[f32], shape: Vec<usize>) -> String {
        let expected: usize = shape.iter().product();
        debug_assert_eq!(expected, values.len(), "declared shape must match the data");
        let key = format!("a{:05}", self.next);
        self.next += 1;
        let offset = self.payload.len();
        self.payload.reserve(values.len() * 4);
        for value in values {
            self.payload.extend_from_slice(&value.to_le_bytes());
        }
        self.arrays.insert(
            key.clone(),
            ArrayEntry {
                dtype: "<f4".to_string(),
                shape,
                offset,
                bytes: values.len() * 4,
            },
        );
        key
    }

    /// Append one `|u1` array, returning its key.
    ///
    /// Shares the `a{:05}` key counter with [`Self::push_f32`], so a censor
    /// plane can never collide with a moment plane and the payload stays one
    /// contiguous, self-describing block.
    pub fn push_u8(&mut self, values: &[u8], shape: Vec<usize>) -> String {
        let expected: usize = shape.iter().product();
        debug_assert_eq!(expected, values.len(), "declared shape must match the data");
        let key = format!("a{:05}", self.next);
        self.next += 1;
        let offset = self.payload.len();
        self.payload.extend_from_slice(values);
        self.arrays.insert(
            key.clone(),
            ArrayEntry {
                dtype: censor_plane::DTYPE.to_string(),
                shape,
                offset,
                bytes: values.len(),
            },
        );
        key
    }

    pub fn finish(self) -> (Vec<u8>, std::collections::BTreeMap<String, ArrayEntry>) {
        (self.payload, self.arrays)
    }
}

/// Serialize a pack to bytes: 64-byte header, JSON metadata, payload.
pub fn encode_pack(meta: &PackMeta, payload: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    let meta_json = serde_json::to_vec(meta)?;
    let mut out = Vec::with_capacity(PACK_HEADER_BYTES + meta_json.len() + payload.len());
    out.extend_from_slice(PACK_MAGIC);
    out.extend_from_slice(&PACK_VERSION.to_le_bytes());
    out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.resize(PACK_HEADER_BYTES, 0);
    out.extend_from_slice(&meta_json);
    out.extend_from_slice(payload);
    Ok(out)
}

/// Read back a pack's metadata and payload, checking every self-describing
/// field the writer promised.
pub fn decode_pack(bytes: &[u8]) -> Result<(PackMeta, Vec<u8>), Box<dyn Error>> {
    if bytes.len() < PACK_HEADER_BYTES {
        return Err(boxed_error(format!(
            "not a radar sweep pack: {} bytes, the header alone is {PACK_HEADER_BYTES}",
            bytes.len()
        )));
    }
    if &bytes[..8] != PACK_MAGIC {
        return Err(boxed_error(format!(
            "not a radar sweep pack: magic {:?}",
            String::from_utf8_lossy(&bytes[..8])
        )));
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != PACK_VERSION {
        return Err(boxed_error(format!(
            "radar sweep pack version {version}, this build reads {PACK_VERSION}"
        )));
    }
    let meta_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let payload_len = u64::from_le_bytes([
        bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
    ]) as usize;
    let meta_end = PACK_HEADER_BYTES + meta_len;
    let payload_end = meta_end + payload_len;
    if payload_end != bytes.len() {
        return Err(boxed_error(format!(
            "truncated radar sweep pack: header declares {payload_end} bytes, file has {}",
            bytes.len()
        )));
    }
    let meta: PackMeta = serde_json::from_slice(&bytes[PACK_HEADER_BYTES..meta_end])?;
    if meta.schema != SWEEPS_SCHEMA
        && meta.schema != SWEEPS_SCHEMA_CENSOR
        && meta.schema != SWEEPS_SCHEMA_ODIM
    {
        return Err(boxed_error(format!(
            "radar sweep pack declares schema {:?}, expected {SWEEPS_SCHEMA:?}, \
             {SWEEPS_SCHEMA_CENSOR:?} or {SWEEPS_SCHEMA_ODIM:?}",
            meta.schema
        )));
    }
    // The schema string and the arrays have to tell the same story.  A `v1`
    // pack carrying censor planes would be read by a `v1` consumer as a
    // plain pack that happens to be larger, and a `v2` pack without them
    // promises a distinction it cannot make; both are refused here rather
    // than left for a consumer to trip over.
    let censored = meta.schema == SWEEPS_SCHEMA_CENSOR || meta.schema == SWEEPS_SCHEMA_ODIM;
    for sweep in &meta.sweeps {
        for moment in &sweep.moments {
            if moment.censor_array.is_some() != censored {
                return Err(boxed_error(format!(
                    "radar sweep pack declares schema {:?} but sweep {} moment {} {} a \
                     censor plane; the schema string and the arrays must agree",
                    meta.schema,
                    sweep.sweep_index,
                    moment.product,
                    if censored { "is missing" } else { "carries" }
                )));
            }
        }
    }
    if meta.params.censor_flags != censored {
        return Err(boxed_error(format!(
            "radar sweep pack declares schema {:?} but params.censor_flags is {}",
            meta.schema, meta.params.censor_flags
        )));
    }
    let payload = bytes[meta_end..payload_end].to_vec();
    let digest = hex_sha256(&payload);
    if digest != meta.content_sha256 {
        return Err(boxed_error(format!(
            "radar sweep pack payload digest mismatch: metadata says {}, bytes hash to {digest}",
            meta.content_sha256
        )));
    }
    check_nyquist_scalars_summarise_their_arrays(&meta, &payload)?;
    Ok((meta, payload))
}

/// Tolerance the scalar must summarise its array to, in m/s.
///
/// The same number the Python sweep reader checks, and both are checking one
/// `f32` widened twice, so any drift beyond rounding is a real defect rather
/// than a formatting artefact.
const NYQUIST_SUMMARY_TOLERANCE_MS: f64 = 1e-3;

/// A sweep's `nyquist_velocity_ms` must be its per-radial array's minimum.
///
/// The scalar is a *reduction* of the array, not a second measurement of the
/// same thing, so the only way a pack can carry both is if reading one back
/// reproduces the other.  Checking it here means a writer defect is caught by
/// the writer's own reader, on the bytes, rather than three lanes downstream
/// by a consumer that had every right to trust the file.
fn check_nyquist_scalars_summarise_their_arrays(
    meta: &PackMeta,
    payload: &[u8],
) -> Result<(), Box<dyn Error>> {
    for sweep in &meta.sweeps {
        let Some(key) = sweep.nyquist_by_radial_array.as_deref() else {
            continue;
        };
        let entry = meta.arrays.get(key).ok_or_else(|| {
            boxed_error(format!(
                "radar sweep pack sweep {} names per-radial Nyquist array {key:?}, which is \
                 not in the array index",
                sweep.sweep_index
            ))
        })?;
        if entry.dtype != "<f4" || entry.shape != vec![sweep.radial_count] {
            return Err(boxed_error(format!(
                "radar sweep pack sweep {} per-radial Nyquist array {key:?} is {} {:?}, \
                 expected \"<f4\" [{}]",
                sweep.sweep_index, entry.dtype, entry.shape, sweep.radial_count
            )));
        }
        if entry.offset + entry.bytes > payload.len() || entry.bytes != sweep.radial_count * 4 {
            return Err(boxed_error(format!(
                "radar sweep pack sweep {} per-radial Nyquist array {key:?} declares \
                 {} bytes at offset {} inside a {}-byte payload",
                sweep.sweep_index,
                entry.bytes,
                entry.offset,
                payload.len()
            )));
        }
        let lowest = payload[entry.offset..entry.offset + entry.bytes]
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .filter(|value| value.is_finite() && *value > 0.0)
            .fold(None::<f32>, |acc, value| {
                Some(acc.map_or(value, |a| a.min(value)))
            });
        match (lowest, sweep.nyquist_velocity_ms) {
            // A present array asserts the originals were kept, so an array
            // with nothing usable in it is a claim with no content.
            (None, _) => {
                return Err(boxed_error(format!(
                    "radar sweep pack sweep {} carries a per-radial Nyquist array with no \
                     usable value in it; a present array asserts the originals were kept",
                    sweep.sweep_index
                )))
            }
            (Some(lowest), None) => {
                return Err(boxed_error(format!(
                    "radar sweep pack sweep {} carries per-radial Nyquist values bottoming \
                     out at {lowest} but declares no nyquist_velocity_ms; the scalar is \
                     defined as that array's minimum",
                    sweep.sweep_index
                )))
            }
            (Some(lowest), Some(scalar)) => {
                if (scalar - lowest as f64).abs() > NYQUIST_SUMMARY_TOLERANCE_MS {
                    return Err(boxed_error(format!(
                        "radar sweep pack sweep {} declares nyquist_velocity_ms {scalar} but \
                         its per-radial array bottoms out at {lowest}; the scalar is defined \
                         as that array's minimum",
                        sweep.sweep_index
                    )));
                }
            }
        }
    }
    Ok(())
}

pub fn write_pack(path: &Path, meta: &PackMeta, payload: &[u8]) -> Result<usize, Box<dyn Error>> {
    let bytes = encode_pack(meta, payload)?;
    atomic_write_bytes(path, &bytes)?;
    Ok(bytes.len())
}

/// Resolve a `--moments` token list against the products the parser knows.
pub fn parse_moment_filter(spec: &str) -> Result<Vec<RadarProduct>, Box<dyn Error>> {
    let mut out = Vec::new();
    for token in spec.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let product = RadarProduct::from_name(&token.to_ascii_uppercase());
        if product == RadarProduct::Unknown {
            return Err(boxed_error(format!(
                "unknown moment {token:?}: expected REF, VEL, SW, ZDR, RHO, PHI or KDP"
            )));
        }
        if !out.contains(&product) {
            out.push(product);
        }
    }
    if out.is_empty() {
        return Err(boxed_error(
            "--moments selected nothing; give at least one of REF, VEL, SW, ZDR, RHO, PHI, KDP"
                .to_string(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal, *well-framed* Archive-II shell: 24-byte volume header
    /// then one LDM block whose payload begins with a bzip2 magic.  It is
    /// not decodable radar data — these tests are about framing.
    fn framed_shell(block_payload: &[u8]) -> Vec<u8> {
        let mut raw = Vec::new();
        raw.extend_from_slice(b"AR2V0006.");
        raw.resize(24, 0);
        raw.extend_from_slice(&(block_payload.len() as i32).to_be_bytes());
        raw.extend_from_slice(block_payload);
        raw
    }

    /// Framing is proved through `read_volume`, the entry point the binary
    /// actually calls, so a gzipped file and a plain one go down exactly
    /// the path the CLI takes.
    fn framing_of(raw: &[u8]) -> Result<Framing, String> {
        read_volume(raw)
            .map(|(_stream, framing)| framing)
            .map_err(|err| err.to_string())
    }

    /// gzip the way the pre-2016 archive does.
    fn gzipped(raw: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(raw).unwrap();
        let out = encoder.finish().unwrap();
        assert_eq!(&out[..2], &[0x1f, 0x8b], "a gzip member starts 1f 8b");
        out
    }

    /// `radials` Message-31 messages laid out as the pre-2016 archive lays
    /// a volume out: straight after the header, no block table.
    fn unframed_shell(radials: usize) -> Vec<u8> {
        let mut raw = Vec::from(&b"AR2V0006."[..]);
        raw.resize(24, 0);
        raw[20..24].copy_from_slice(b"KTLX");
        for _ in 0..radials {
            let body = 600usize;
            let mut message = vec![0u8; 12];
            message.extend_from_slice(&(((body - 12) / 2) as u16).to_be_bytes());
            message.push(0);
            message.push(31);
            message.resize(body, 0);
            raw.extend_from_slice(&message);
        }
        raw
    }

    #[test]
    fn framing_accepts_a_whole_ldm_volume() {
        let raw = framed_shell(b"BZh9payload-bytes");
        let framing = framing_of(&raw).unwrap();
        assert_eq!(framing.layout, layout::LDM_BZIP2);
        assert!(!framing.gzip_wrapped);
        assert_eq!(framing.block_count, 1);
        assert_eq!(framing.bzip2_block_count, 1);
        assert_eq!(framing.bytes, raw.len());
        assert_eq!(framing.source_bytes, raw.len());
        assert!(framing.magic.starts_with("AR2V"));
    }

    #[test]
    fn framing_accepts_the_pre_2016_shape_plain_and_gzipped() {
        // The archive stores roughly 2011-2016 gzipped and everything after
        // it plain.  A front door that reads one spelling and refuses the
        // other cannot see half the record.
        let plain = unframed_shell(3);
        let framing = framing_of(&plain).unwrap();
        assert_eq!(framing.layout, layout::UNCOMPRESSED_MESSAGES);
        assert!(!framing.gzip_wrapped);
        assert_eq!(framing.message_count, 3);
        assert_eq!(framing.block_count, 0);
        assert_eq!(framing.bytes, plain.len());
        assert_eq!(framing.source_bytes, plain.len());

        let wrapped = gzipped(&plain);
        assert!(wrapped.len() < plain.len(), "the fixture must really compress");
        let framing = framing_of(&wrapped).unwrap();
        assert_eq!(framing.layout, layout::UNCOMPRESSED_MESSAGES);
        assert!(framing.gzip_wrapped, "the wrapper must be recorded");
        assert_eq!(framing.message_count, 3);
        // The expansion is what was validated; the file is what arrived,
        // and it is the file the listing's size and the sha256 describe.
        assert_eq!(framing.bytes, plain.len());
        assert_eq!(framing.source_bytes, wrapped.len());

        // The wrapper and the inner layout are independent: a gzipped LDM
        // volume is equally readable, and neither implies the other.
        let framed = framed_shell(b"BZh9payload-bytes");
        let framing = framing_of(&gzipped(&framed)).unwrap();
        assert_eq!(framing.layout, layout::LDM_BZIP2);
        assert!(framing.gzip_wrapped);
    }

    #[test]
    fn framing_refuses_a_truncated_block() {
        let mut raw = framed_shell(b"BZh9payload-bytes");
        raw.truncate(raw.len() - 5);
        let err = framing_of(&raw).unwrap_err();
        assert!(err.contains("truncated"), "{err}");
        assert!(err.contains("only"), "{err}");
    }

    #[test]
    fn framing_refuses_a_dangling_length_word() {
        let mut raw = framed_shell(b"BZh9payload-bytes");
        raw.extend_from_slice(&[0u8, 0u8]);
        let err = framing_of(&raw).unwrap_err();
        assert!(err.contains("cannot hold a 4-byte LDM block length"), "{err}");
    }

    #[test]
    fn framing_refuses_a_pre_2016_stream_that_arrived_short() {
        // The unframed layout's completeness argument: the messages must
        // tile the file exactly.  Three cuts, three refusals.
        let mut raw = unframed_shell(3);
        raw.truncate(raw.len() - 300);
        assert!(framing_of(&raw).unwrap_err().contains("truncated"));

        let mut raw = unframed_shell(3);
        raw.truncate(raw.len() - 4);
        assert!(framing_of(&raw).unwrap_err().contains("truncated"));

        let mut raw = unframed_shell(2);
        raw.extend_from_slice(&[0u8; 6]);
        let err = framing_of(&raw).unwrap_err();
        assert!(err.contains("cannot hold a message header"), "{err}");
    }

    #[test]
    fn framing_refuses_a_pre_2016_stream_with_no_radial() {
        // Messages that tile perfectly, none of them Message-31: framed
        // cleanly and carrying nothing to observe.
        let mut raw = Vec::from(&b"AR2V0006."[..]);
        raw.resize(24, 0);
        raw.resize(24 + 2432 * 2, 0);
        raw[24 + 15] = 2;
        raw[24 + 2432 + 15] = 3;
        let err = framing_of(&raw).unwrap_err();
        assert!(err.contains("none is a Message-31 radial"), "{err}");
    }

    #[test]
    fn framing_refuses_non_archive2_and_stubs() {
        let err = framing_of(&vec![7u8; 64]).unwrap_err();
        assert!(err.contains("not a Level-II volume"), "{err}");

        // A gzip member holding something that is not a volume is refused
        // for what it holds, not for being gzipped.
        let err = framing_of(&gzipped(&vec![7u8; 64])).unwrap_err();
        assert!(err.contains("not a Level-II volume"), "{err}");

        // A gzip wrapper cut short is refused, at several cut points.  Some
        // of these fail in the inflate itself and some decode to a clean
        // prefix and are caught by the trailer's ISIZE; both are refusals,
        // and the second is the one that would otherwise have published the
        // first N radials of a volume as though it were the whole thing.
        let whole = gzipped(&unframed_shell(4));
        for divisor in [2usize, 3, 4] {
            let mut broken = whole.clone();
            broken.truncate(broken.len() / divisor);
            let err = framing_of(&broken).unwrap_err();
            assert!(
                err.contains("will not expand") || err.contains("incomplete"),
                "cut to 1/{divisor}: {err}"
            );
        }
        // Losing only the trailer is the case flate2 does not report.
        let mut trailerless = whole.clone();
        trailerless.truncate(whole.len() - 8);
        let err = framing_of(&trailerless).unwrap_err();
        assert!(
            err.contains("will not expand") || err.contains("incomplete"),
            "{err}"
        );
        // And the whole member still passes, so the check is not blanket.
        assert!(framing_of(&whole).is_ok());

        let err = framing_of(b"AR2V000").unwrap_err();
        assert!(err.contains("24-byte volume header"), "{err}");
    }

    #[test]
    fn the_two_crates_classify_the_same_bytes_alike() {
        // The framing record says which shape it validated and the decoder
        // decides the same question independently.  If they ever disagree
        // the record describes a volume that was not the one parsed, so the
        // rule is pinned across the seam rather than assumed.
        let cases: Vec<Vec<u8>> = vec![
            framed_shell(b"BZh9payload-bytes"),
            framed_shell(b"not-a-bzip2-stream"),
            unframed_shell(1),
            unframed_shell(4),
            {
                let mut header_only = Vec::from(&b"AR2V0006."[..]);
                header_only.resize(24, 0);
                header_only
            },
            b"AR2V000".to_vec(),
            vec![7u8; 64],
        ];
        for raw in cases {
            assert_eq!(
                is_ldm_framed(&raw),
                Level2File::is_ldm_framed(&raw),
                "the two crates disagree about {} bytes beginning {:?}",
                raw.len(),
                &raw[..raw.len().min(8)]
            );
        }
    }

    #[test]
    fn framing_refuses_an_ldm_table_that_declares_nothing_usable() {
        // A bzip2 first block proves the LDM shape, so a later bad block
        // stays on that path and keeps its original diagnosis.
        let mut raw = framed_shell(b"BZh9payload-bytes");
        raw.extend_from_slice(&0i32.to_be_bytes());
        let err = framing_of(&raw).unwrap_err();
        assert!(err.contains("declares zero bytes"), "{err}");

        // A header with nothing after it is neither shape.
        let mut header_only = Vec::from(&b"AR2V0006."[..]);
        header_only.resize(24, 0);
        let err = framing_of(&header_only).unwrap_err();
        assert!(err.contains("none is a Message-31 radial"), "{err}");
    }

    #[test]
    fn a_block_table_carrying_no_bzip2_is_refused_as_the_shape_it_is() {
        // This used to be "N LDM blocks, none of them a bzip2 stream".  That
        // refusal is gone because it cannot fire: the classifier only sends
        // a volume down the LDM path when the *first* block opens `BZh`, so
        // by then at least one block is bzip2.  A table of plain blocks is
        // not a damaged LDM volume, it is bytes that are not an LDM volume,
        // and it is now refused as a message stream that does not tile.
        // What must not change is that it is refused.
        for payload in [
            &b"plain-uncompressed-bytes"[..],
            &b"BZ"[..],
            &b"gzip\x1f\x8bnot-bzip2"[..],
        ] {
            let err = framing_of(&framed_shell(payload)).unwrap_err();
            assert!(
                err.contains("truncated Level-II volume")
                    || err.contains("none is a Message-31 radial"),
                "{payload:?}: {err}"
            );
        }
    }

    fn sample_meta(payload: &[u8], arrays: std::collections::BTreeMap<String, ArrayEntry>) -> PackMeta {
        PackMeta {
            schema: SWEEPS_SCHEMA.to_string(),
            status: "READY".to_string(),
            site: SiteEntry {
                id: "KTLX".to_string(),
                name: "Oklahoma City, OK".to_string(),
                lat_deg: 35.3331,
                lon_deg: -97.2778,
                alt_m: 370.0,
                source: "wx-radar-site-table".to_string(),
            },
            volume: VolumeEntry {
                file: "KTLX20230520_200356_V06".to_string(),
                bytes: 42,
                sha256: hex_sha256(b"volume"),
                station_id: "KTLX".to_string(),
                valid_time: "2023-05-20T20:03:56Z".to_string(),
                volume_date: 19497,
                volume_time_ms: 72236000,
                framing: Some(Framing {
                    magic: "AR2V0006".to_string(),
                    layout: layout::LDM_BZIP2.to_string(),
                    gzip_wrapped: false,
                    block_count: 3,
                    bzip2_block_count: 3,
                    message_count: 0,
                    bytes: 42,
                    source_bytes: 42,
                }),
                assembled: None,
            },
            params: DecodeParams {
                moments: vec!["REF".to_string(), "VEL".to_string()],
                max_range_km: 300.0,
                max_elevation_deg: 20.0,
                censor_flags: false,
            },
            sweeps: Vec::new(),
            arrays,
            payload_bytes: payload.len(),
            content_sha256: hex_sha256(payload),
            dropped_sweeps: 0,
            dropped_moments: 0,
            trimmed_gates: 0,
        }
    }

    #[test]
    fn pack_round_trips_through_bytes() {
        let mut builder = PayloadBuilder::new();
        let azimuth = builder.push_f32(&[0.0, 1.0, 2.0], vec![3]);
        let data = builder.push_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]);
        let (payload, arrays) = builder.finish();
        assert_eq!(arrays[&azimuth].offset, 0);
        assert_eq!(arrays[&data].offset, 12);
        assert_eq!(arrays[&data].shape, vec![3, 2]);

        let meta = sample_meta(&payload, arrays);
        let bytes = encode_pack(&meta, &payload).unwrap();
        let (read_meta, read_payload) = decode_pack(&bytes).unwrap();
        assert_eq!(read_meta.schema, SWEEPS_SCHEMA);
        assert_eq!(read_meta.site.id, "KTLX");
        assert_eq!(read_payload, payload);
        assert_eq!(read_meta.content_sha256, hex_sha256(&payload));
    }

    #[test]
    fn pack_reader_refuses_truncation_bad_magic_and_digest_drift() {
        let mut builder = PayloadBuilder::new();
        builder.push_f32(&[1.0, 2.0], vec![2]);
        let (payload, arrays) = builder.finish();
        let meta = sample_meta(&payload, arrays);
        let bytes = encode_pack(&meta, &payload).unwrap();

        let mut truncated = bytes.clone();
        truncated.truncate(bytes.len() - 1);
        assert!(
            decode_pack(&truncated).unwrap_err().to_string().contains("truncated"),
        );

        let mut wrong_magic = bytes.clone();
        wrong_magic[..8].copy_from_slice(b"NOTAPACK");
        assert!(
            decode_pack(&wrong_magic).unwrap_err().to_string().contains("magic"),
        );

        let mut wrong_version = bytes.clone();
        wrong_version[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(
            decode_pack(&wrong_version).unwrap_err().to_string().contains("version 99"),
        );

        // Flip a payload byte: the digest in metadata must catch it.
        let mut corrupt = bytes.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xff;
        assert!(
            decode_pack(&corrupt).unwrap_err().to_string().contains("digest mismatch"),
        );
    }

    #[test]
    fn moment_filter_resolves_tokens_and_refuses_junk() {
        let products = parse_moment_filter("ref, vel ,REF").unwrap();
        assert_eq!(products, vec![RadarProduct::Reflectivity, RadarProduct::Velocity]);
        assert!(parse_moment_filter("REF,BOGUS").is_err());
        assert!(parse_moment_filter(" , ").is_err());
    }
}
