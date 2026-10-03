//! Level II message walker and typed message bodies.
//!
//! References are to the RDA/RPG ICD, document 2620002AA (Build 24.0,
//! 19 August 2025, NOAA Radar Operations Center), which is the latest
//! published revision.
//!
//! Archive II record bytes are a sequence of frames. Each frame starts with a
//! 12-byte communications-manager (CTM) header, then the 16-byte message
//! header of Table II, then the message body:
//!
//! - Messages 29 and 31 are variable length: the frame is the CTM header plus
//!   `size_halfwords * 2` bytes.
//! - A size field of 65535 (Table II notes 6 and 7) means bytes 12-15 of the
//!   message header carry the message size in bytes, header included; the
//!   frame is the CTM header plus that many bytes and the message is one
//!   segment.
//! - Every other message occupies a fixed 2432-byte frame (12-byte CTM header,
//!   up to 2416 message bytes, padding). A size of zero marks an empty frame,
//!   common in the 134-frame metadata record.
//! - Messages too large for one frame (for example 13, 15 and 18) are split
//!   into segments carrying the same message type with `segments` and
//!   `segment_number` set; [`RawMessages`] concatenates the segment bodies
//!   and yields the message once, and [`MessageWalker`] decodes it.
//!
//! The walker is for metadata and inspection. Radial decoding keeps its own
//! fast path in [`crate::read_volume_from_bytes`]; message 1 and 29 bodies
//! are yielded as [`MessageBody::Unparsed`], and message 31 bodies are decoded
//! completely by [`msg31_blocks`].

pub mod adaptation;
pub mod bypass_map;
pub mod clutter_censor;
pub mod clutter_filter_map;
pub mod console;
pub mod control;
pub mod loopback;
pub mod msg31_blocks;
pub mod performance;
pub mod prf;
pub mod rda_log;
pub mod rda_status;
pub mod request;
pub mod vcp;

use std::borrow::Cow;
use std::collections::VecDeque;
use std::io::Read;

use chrono::{DateTime, Utc};
use recast_radar_core::bounded_read::{self, MAX_DECODED_RADAR_BYTES};

use crate::{
    CONTROL_WORD_LEN, MESSAGE_HEADER_LEN, MessageHeader, NexradError, RECORD_BYTES, Result,
    VOLUME_HEADER_LEN,
};

/// Number of fixed frames in the Archive II metadata record.
pub const METADATA_RECORD_FRAMES: usize = 134;

/// Size-field value announcing that header bytes 12-15 hold the message size
/// in bytes (Table II note 6).
pub const EXTENDED_SIZE_SENTINEL: u16 = 0xFFFF;

/// Name of a message type per Table I, or `None` for codes Table I does not
/// define.
pub fn message_type_name(message_type: u8) -> Option<&'static str> {
    Some(match message_type {
        1 => "Digital Radar Data",
        2 => "RDA Status Data",
        3 => "Performance/Maintenance Data",
        4 | 10 => "Console Message",
        5 | 7 => "Volume Coverage Pattern",
        6 => "RDA Control Commands",
        8 => "Clutter Censor Zones",
        9 => "Request for Data",
        11 | 12 => "Loop Back Test",
        13 => "Clutter Filter Bypass Map",
        14 => "Spare",
        15 => "Clutter Filter Map",
        16 | 17 | 24 | 25 | 26 => "Reserved/FAA RMS Only",
        18 => "RDA Adaptation Data",
        20..=23 | 29 => "Reserved",
        31 => "Digital Radar Data Generic Format",
        32 => "RDA PRF Data",
        33 => "RDA Log Data",
        _ => return None,
    })
}

impl MessageHeader {
    /// True when the size field is 65535 and bytes 12-15 hold the message
    /// size in bytes (Table II notes 6 and 7).
    pub fn has_extended_size(&self) -> bool {
        self.size_halfwords == EXTENDED_SIZE_SENTINEL
    }

    /// Length of this message (or message segment) in bytes, including the
    /// 16-byte message header.
    pub fn message_len(&self) -> usize {
        if self.has_extended_size() {
            (usize::from(self.segments) << 16) | usize::from(self.segment_number)
        } else {
            usize::from(self.size_halfwords) * 2
        }
    }

    /// Number of segments in the whole message: 1 for extended-size messages,
    /// whose segment fields carry the size instead.
    pub fn segment_count(&self) -> u16 {
        if self.has_extended_size() {
            1
        } else {
            self.segments
        }
    }

    /// True for messages framed by their own length rather than a fixed
    /// 2432-byte frame: messages 29 and 31, and any extended-size message.
    pub fn is_variable_length(&self) -> bool {
        self.has_extended_size() || matches!(self.message_type, 29 | 31)
    }

    /// Header generation time from the modified Julian date and milliseconds
    /// of day (Table II note 2: 1 January 1970 is day 1).
    pub fn timestamp(&self) -> DateTime<Utc> {
        crate::nexrad_date_ms_to_datetime(u32::from(self.date), self.milliseconds)
    }
}

/// Decoded body of one Level II message.
///
/// Variants other than [`MessageBody::Unparsed`] exist for every message
/// table in the ICD. A variant is produced once its decoder is implemented;
/// until then (and for radial data, reserved and unknown types) the walker
/// yields the raw body bytes as [`MessageBody::Unparsed`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum MessageBody<'a> {
    /// Message 2, RDA Status Data (Table IV).
    RdaStatus(rda_status::RdaStatus),
    /// Message 3, Performance/Maintenance Data (Table V).
    Performance(Box<performance::PerformanceMaintenance>),
    /// Messages 4 (RDA to RPG) and 10 (RPG to RDA), Console Message (Table VI).
    Console(console::ConsoleMessage),
    /// Messages 5 (RDA to RPG) and 7 (RPG to RDA), Volume Coverage Pattern
    /// (Table XI).
    Vcp(vcp::VolumeCoveragePattern),
    /// Message 6, RDA Control Commands (Table X).
    ControlCommands(control::RdaControlCommands),
    /// Message 8, Clutter Censor Zones (Table XII).
    ClutterCensorZones(clutter_censor::ClutterCensorZones),
    /// Message 9, Request for Data (Table XIII).
    RequestForData(request::RequestForData),
    /// Messages 11 (RDA to RPG) and 12 (RPG to RDA), Loop Back Test
    /// (Table VIII).
    Loopback(loopback::LoopbackTest),
    /// Message 13, Clutter Filter Bypass Map (Table IX).
    BypassMap(bypass_map::ClutterFilterBypassMap),
    /// Message 15, Clutter Filter Map (Table XIV).
    ClutterFilterMap(clutter_filter_map::ClutterFilterMap),
    /// Message 18, RDA Adaptation Data (Table XV).
    Adaptation(Box<adaptation::RdaAdaptationData>),
    /// Message 31, Digital Radar Data Generic Format (Table XVII), boxed
    /// because it is much larger than the other variants.
    DigitalRadarDataGeneric(Box<msg31_blocks::DigitalRadarDataGeneric<'a>>),
    /// Message 32, RDA PRF Data (Table XVIII).
    Prf(prf::RdaPrfData),
    /// Message 33, RDA Log Data (Table XVIV).
    RdaLog(rda_log::RdaLogData),
    /// Body bytes (after the 16-byte message header, reassembled across
    /// segments) of a message without a typed decoder.
    Unparsed(Cow<'a, [u8]>),
}

/// Route a complete message body to its table decoder. Messages 2, 3 and 18
/// also need the header's RDA channel byte to tell legacy and Open RDA
/// layouts apart.
fn decode_body<'a>(header: &MessageHeader, body: Cow<'a, [u8]>) -> Result<MessageBody<'a>> {
    match header.message_type {
        2 => rda_status::message_body(header, body),
        3 => performance::message_body(header, body),
        4 | 10 => console::message_body(body),
        5 | 7 => vcp::message_body(body),
        6 => control::message_body(body),
        8 => clutter_censor::message_body(body),
        9 => request::message_body(body),
        11 | 12 => loopback::message_body(body),
        13 => bypass_map::message_body(body),
        15 => clutter_filter_map::message_body(body),
        18 => adaptation::message_body(header, body),
        31 => msg31_blocks::message_body(body),
        32 => prf::message_body(body),
        33 => rda_log::message_body(body),
        _ => Ok(MessageBody::Unparsed(body)),
    }
}

/// One message from [`RawMessages`], before table decoding.
#[derive(Clone, Debug, PartialEq)]
pub struct RawMessage<'a> {
    /// Message header; for a segmented message, the first segment's header,
    /// unchanged.
    pub header: MessageHeader,
    /// Byte offset of the (first) message header within the walked input.
    pub offset: usize,
    /// Number of frames joined into this message (1 unless segmented).
    pub frames: usize,
    /// Body bytes after the 16-byte message header, concatenated across
    /// segments.
    pub body: Cow<'a, [u8]>,
}

impl<'a> RawMessage<'a> {
    /// Decode the body with its table decoder. Errors are wrapped in
    /// [`NexradError::InvalidMessage`] naming the message type and offset.
    pub fn decode(self) -> Result<(MessageHeader, MessageBody<'a>)> {
        let message_type = self.header.message_type;
        match decode_body(&self.header, self.body) {
            Ok(body) => Ok((self.header, body)),
            Err(error) => Err(NexradError::InvalidMessage {
                offset: self.offset,
                reason: format!("message type {message_type}: {error}"),
            }),
        }
    }
}

/// Iterator over the messages in decompressed Archive II record bytes, with
/// segments reassembled and bodies left undecoded. [`MessageWalker`] decodes
/// the bodies.
///
/// The input starts at a frame boundary: the bytes after the 24-byte volume
/// header, one decompressed LDM record, or several records concatenated
/// (records always end on a frame boundary). See the module documentation for
/// the framing rules.
///
/// Items are `Ok` per message, or `Err` for a problem with one message, after
/// which walking continues:
///
/// - a fixed-frame message declaring a size smaller than its header or larger
///   than a frame (the frame is skipped);
/// - a segmented message cut short by a different message or by the end of
///   the input, or a run of continuation segments whose first segment is
///   missing (one error per run).
///
/// Truncation (a partial frame header, a variable-length message running past
/// the end of the input) and a variable-length message declaring a size
/// smaller than its header yield one `Err` and end the walk, because the next
/// frame boundary is unknown.
///
/// Reassembly: a frame numbered 1 whose segment count is above 1 starts a
/// message; frames of the same type numbered 2, 3, ... in a row append to it;
/// the message is complete at the frame whose number reaches its own segment
/// count. Segments must be consecutive, as they are in every Archive II file
/// in the corpus. The segment count is not required to agree between
/// segments: Build 10 metadata records (for example KPAH 2008-04-15) carry a
/// Message 15 whose first segment says 5 segments and whose other 76 say 77,
/// all with one generation time. Stale frames left in the fixed metadata
/// record by an earlier, longer message (numbered past the new message's end,
/// sometimes with the type zeroed) form orphan runs.
pub struct RawMessages<'a> {
    bytes: &'a [u8],
    cursor: usize,
    frames: usize,
    empty_frames: usize,
    pending: Option<PendingMessage>,
    orphans: Option<OrphanSegments>,
    queue: VecDeque<Result<RawMessage<'a>>>,
    finished: bool,
}

/// A segmented message being reassembled.
struct PendingMessage {
    /// Header of the first segment.
    header: MessageHeader,
    offset: usize,
    frames: usize,
    next_segment: u16,
    /// Segment count declared by the latest segment.
    last_segments: u16,
    body: Vec<u8>,
}

/// A run of consecutively numbered continuation segments with no first
/// segment.
struct OrphanSegments {
    message_type: u8,
    segments: u16,
    first_segment: u16,
    last_segment: u16,
    offset: usize,
}

impl<'a> RawMessages<'a> {
    /// Walk record bytes that start at a frame boundary.
    pub fn new(records: &'a [u8]) -> Self {
        Self {
            bytes: records,
            cursor: 0,
            frames: 0,
            empty_frames: 0,
            pending: None,
            orphans: None,
            queue: VecDeque::new(),
            finished: false,
        }
    }

    /// Walk record bytes that may start with the 24-byte Archive II volume
    /// header (see [`volume_header_len`]), skipping it. Offsets are relative
    /// to the bytes after the header.
    pub fn skipping_volume_header(bytes: &'a [u8]) -> Self {
        Self::new(&bytes[volume_header_len(bytes)..])
    }

    /// Byte offset, within the walked input, of the next frame to read.
    pub fn position(&self) -> usize {
        self.cursor
    }

    /// Frames read so far, including empty frames and each segment.
    pub fn frame_count(&self) -> usize {
        self.frames
    }

    /// Empty (size zero) fixed frames read so far.
    pub fn empty_frame_count(&self) -> usize {
        self.empty_frames
    }

    fn read_frame(&mut self) {
        let frame_offset = self.cursor;
        let remaining = self.bytes.len() - frame_offset;
        let header_offset = frame_offset + CONTROL_WORD_LEN;
        if remaining < CONTROL_WORD_LEN + MESSAGE_HEADER_LEN {
            self.finish(Some(NexradError::Truncated {
                what: "message frame header",
                offset: frame_offset,
                needed: CONTROL_WORD_LEN + MESSAGE_HEADER_LEN,
                available: remaining,
            }));
            return;
        }
        let header = crate::parse_message_header_bytes(
            &self.bytes[header_offset..header_offset + MESSAGE_HEADER_LEN],
        );
        self.frames += 1;

        if header.size_halfwords == 0 {
            self.empty_frames += 1;
            self.advance_fixed(frame_offset);
            return;
        }

        let message_len = header.message_len();
        let variable = header.is_variable_length();
        if message_len < MESSAGE_HEADER_LEN {
            let error = NexradError::InvalidMessage {
                offset: header_offset,
                reason: format!(
                    "message type {} declares {message_len} bytes, less than its {MESSAGE_HEADER_LEN}-byte header",
                    header.message_type
                ),
            };
            if variable {
                self.finish(Some(error));
            } else {
                self.queue.push_back(Err(error));
                self.advance_fixed(frame_offset);
            }
            return;
        }
        if !variable && CONTROL_WORD_LEN + message_len > RECORD_BYTES {
            self.queue.push_back(Err(NexradError::InvalidMessage {
                offset: header_offset,
                reason: format!(
                    "message type {} declares {message_len} bytes, more than a {RECORD_BYTES}-byte frame holds",
                    header.message_type
                ),
            }));
            self.advance_fixed(frame_offset);
            return;
        }
        let available = remaining - CONTROL_WORD_LEN;
        if message_len > available {
            self.finish(Some(NexradError::Truncated {
                what: "message body",
                offset: header_offset,
                needed: message_len,
                available,
            }));
            return;
        }

        let body_start = header_offset + MESSAGE_HEADER_LEN;
        let body = if variable {
            &self.bytes[body_start..header_offset + message_len]
        } else {
            // A Message 5 or 7 may run past its declared size within the
            // frame (vcp::fixed_frame_body_len).
            let frame_end = frame_offset
                .saturating_add(RECORD_BYTES)
                .min(self.bytes.len());
            let frame_body = &self.bytes[body_start..frame_end];
            let len = vcp::fixed_frame_body_len(
                header.message_type,
                message_len - MESSAGE_HEADER_LEN,
                frame_body,
            );
            &frame_body[..len]
        };
        if variable {
            self.cursor = header_offset + message_len;
        } else {
            self.advance_fixed(frame_offset);
        }

        if variable || header.segments <= 1 {
            self.flush_partial();
            self.queue.push_back(Ok(RawMessage {
                header,
                offset: header_offset,
                frames: 1,
                body: Cow::Borrowed(body),
            }));
        } else {
            self.add_segment(header, header_offset, body);
        }
    }

    /// Move to the next fixed frame; a final frame may end early when its
    /// message fits but the padding does not.
    fn advance_fixed(&mut self, frame_offset: usize) {
        self.cursor = frame_offset
            .saturating_add(RECORD_BYTES)
            .min(self.bytes.len());
    }

    fn add_segment(&mut self, header: MessageHeader, offset: usize, body: &[u8]) {
        let segment_number = header.segment_number;
        let segments = header.segments;
        let continues_pending = self.pending.as_ref().is_some_and(|pending| {
            pending.header.message_type == header.message_type
                && pending.next_segment == segment_number
        });
        if continues_pending {
            if let Some(pending) = self.pending.as_mut() {
                pending.body.extend_from_slice(body);
                pending.frames += 1;
                pending.next_segment = pending.next_segment.saturating_add(1);
                pending.last_segments = segments;
            }
        } else if segment_number == 1 {
            self.flush_partial();
            self.pending = Some(PendingMessage {
                header,
                offset,
                frames: 1,
                next_segment: 2,
                last_segments: segments,
                body: body.to_vec(),
            });
        } else {
            self.flush_pending();
            let extends_orphans = self.orphans.as_ref().is_some_and(|run| {
                run.message_type == header.message_type
                    && run.last_segment.checked_add(1) == Some(segment_number)
            });
            if extends_orphans {
                if let Some(run) = self.orphans.as_mut() {
                    run.last_segment = segment_number;
                    run.segments = segments;
                }
            } else {
                self.flush_orphans();
                self.orphans = Some(OrphanSegments {
                    message_type: header.message_type,
                    segments,
                    first_segment: segment_number,
                    last_segment: segment_number,
                    offset,
                });
            }
            return;
        }

        if segment_number >= segments
            && let Some(pending) = self.pending.take()
        {
            self.queue.push_back(Ok(RawMessage {
                header: pending.header,
                offset: pending.offset,
                frames: pending.frames,
                body: Cow::Owned(pending.body),
            }));
        }
    }

    /// Report any incomplete segmented message and orphan segment run.
    fn flush_partial(&mut self) {
        self.flush_pending();
        self.flush_orphans();
    }

    fn flush_pending(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.queue.push_back(Err(NexradError::InvalidMessage {
                offset: pending.offset,
                reason: format!(
                    "segmented message type {} ended after {} of {} segments",
                    pending.header.message_type, pending.frames, pending.last_segments
                ),
            }));
        }
    }

    fn flush_orphans(&mut self) {
        if let Some(run) = self.orphans.take() {
            self.queue.push_back(Err(NexradError::InvalidMessage {
                offset: run.offset,
                reason: format!(
                    "message type {} segments {}..={} of {} have no first segment",
                    run.message_type, run.first_segment, run.last_segment, run.segments
                ),
            }));
        }
    }

    fn finish(&mut self, error: Option<NexradError>) {
        self.flush_partial();
        if let Some(error) = error {
            self.queue.push_back(Err(error));
        }
        self.cursor = self.bytes.len();
        self.finished = true;
    }
}

impl<'a> Iterator for RawMessages<'a> {
    type Item = Result<RawMessage<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.queue.pop_front() {
                return Some(item);
            }
            if self.finished {
                return None;
            }
            if self.cursor >= self.bytes.len() {
                self.finish(None);
            } else {
                self.read_frame();
            }
        }
    }
}

/// Iterator over decoded messages: [`RawMessages`] with each body passed
/// through its table decoder ([`RawMessage::decode`]).
///
/// Yields `(header, body)` per message. A body its decoder rejects yields
/// [`NexradError::InvalidMessage`] naming the message type and offset, and
/// walking continues; framing errors behave as in [`RawMessages`].
pub struct MessageWalker<'a> {
    raw: RawMessages<'a>,
}

impl<'a> MessageWalker<'a> {
    /// Walk record bytes that start at a frame boundary.
    pub fn new(records: &'a [u8]) -> Self {
        Self {
            raw: RawMessages::new(records),
        }
    }

    /// Walk record bytes that may start with the 24-byte Archive II volume
    /// header, skipping it.
    pub fn skipping_volume_header(bytes: &'a [u8]) -> Self {
        Self {
            raw: RawMessages::skipping_volume_header(bytes),
        }
    }

    /// The underlying raw walker, for its position and frame counters.
    pub fn raw(&self) -> &RawMessages<'a> {
        &self.raw
    }
}

impl<'a> From<RawMessages<'a>> for MessageWalker<'a> {
    fn from(raw: RawMessages<'a>) -> Self {
        Self { raw }
    }
}

impl<'a> Iterator for MessageWalker<'a> {
    type Item = Result<(MessageHeader, MessageBody<'a>)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.raw
            .next()
            .map(|item| item.and_then(RawMessage::decode))
    }
}

/// Length of the Archive II volume header at the start of `bytes`: 24 when
/// they start with `AR2V` or `ARCHIVE2`, otherwise 0 (real-time intermediate
/// chunks, model-data files and bare records have no volume header).
pub fn volume_header_len(bytes: &[u8]) -> usize {
    if bytes.len() >= VOLUME_HEADER_LEN && crate::starts_with_volume_header(bytes) {
        VOLUME_HEADER_LEN
    } else {
        0
    }
}

/// Decompressed record bytes of a Level II file or real-time chunk, with the
/// volume header (if any) removed and every record concatenated.
///
/// Accepts whole-file gzip or bzip2, LDM bzip2 records (with or without a
/// volume header), and uncompressed records. Output is bounded by
/// [`MAX_DECODED_RADAR_BYTES`].
pub fn record_bytes(raw: &[u8]) -> Result<Cow<'_, [u8]>> {
    let unwrapped = match expand_whole_file(raw, WHOLE_FILE_PAYLOAD)? {
        Some(expanded) => Cow::Owned(expanded),
        None => Cow::Borrowed(raw),
    };
    let records = &unwrapped[volume_header_len(&unwrapped)..];
    let Some(blocks) = ldm_blocks(records)? else {
        return Ok(match unwrapped {
            Cow::Borrowed(bytes) => Cow::Borrowed(&bytes[volume_header_len(bytes)..]),
            Cow::Owned(mut bytes) => {
                bytes.drain(..volume_header_len(&bytes));
                Cow::Owned(bytes)
            }
        });
    };
    let mut output = Vec::new();
    let mut decoded = Vec::new();
    for block in blocks {
        crate::decompress_bzip_block_into(block, &mut decoded)?;
        if output.len() + decoded.len() > MAX_DECODED_RADAR_BYTES {
            return Err(NexradError::Compression(format!(
                "LDM records expand beyond the {MAX_DECODED_RADAR_BYTES}-byte limit"
            )));
        }
        output.extend_from_slice(&decoded);
    }
    Ok(Cow::Owned(output))
}

/// Decompressed bytes of the Archive II metadata record: the first LDM record
/// of an LDM-compressed file, otherwise the first
/// [`METADATA_RECORD_FRAMES`] fixed frames after the volume header (or all
/// records when the file is shorter).
///
/// Files from before the metadata record existed (ARCHIVE2 headers, Message 1
/// only) return their first frames, which then hold radials; the walker
/// yields those as unparsed Message 1 bodies. Whole-file gzip and bzip2
/// inputs are only expanded as far as the metadata record: for raw records
/// its frames, for LDM records the first record (the rest of the file is
/// neither expanded nor checked, so a wrapper damaged past the metadata
/// record still yields it). Of LDM records only the first is framed and
/// decoded, so a file cut short or damaged after it yields it too.
pub fn metadata_record(raw: &[u8]) -> Result<Cow<'_, [u8]>> {
    let metadata_len = METADATA_RECORD_FRAMES * RECORD_BYTES;
    if let Some(prefix) = expand_whole_file_prefix(raw, VOLUME_HEADER_LEN + metadata_len)? {
        let mut expanded = prefix.bytes;
        let header_len = volume_header_len(&expanded);
        if !starts_with_ldm_record(&expanded[header_len..]) {
            expanded.drain(..header_len);
            expanded.truncate(metadata_len);
            return Ok(Cow::Owned(expanded));
        }
        // LDM records inside a whole-file wrapper. The metadata record
        // compresses to a small part of the prefix, so its LDM record is
        // almost always inside it: keep exactly that record. Otherwise
        // expand the wrapper whole, as before the prefix decode.
        let first_end = first_ldm_record_end(&expanded[header_len..])
            .and_then(|end| end.checked_add(header_len));
        let unwrapped = match first_end {
            _ if prefix.complete => expanded,
            Some(end) if end <= expanded.len() => {
                expanded.truncate(end);
                expanded
            }
            _ => {
                let context = if raw.starts_with(b"BZh") {
                    WHOLE_FILE_METADATA
                } else {
                    WHOLE_FILE_PAYLOAD
                };
                expand_whole_file(raw, context)?.ok_or_else(|| {
                    NexradError::Compression("whole-file wrapper disappeared".to_owned())
                })?
            }
        };
        let records = &unwrapped[volume_header_len(&unwrapped)..];
        return Ok(Cow::Owned(first_ldm_record(records)?));
    }

    let records = &raw[volume_header_len(raw)..];
    if starts_with_ldm_record(records) {
        return Ok(Cow::Owned(first_ldm_record(records)?));
    }
    Ok(Cow::Borrowed(&records[..records.len().min(metadata_len)]))
}

/// Decompress the first of the LDM records in `records`. Only that record
/// is framed, so a file cut short or damaged in a later record still
/// yields it.
fn first_ldm_record(records: &[u8]) -> Result<Vec<u8>> {
    let first_end =
        first_ldm_record_end(records).map_or(records.len(), |end| end.min(records.len()));
    let mut decoded = Vec::new();
    if let Some([first, ..]) = ldm_blocks(&records[..first_end])?.as_deref() {
        crate::decompress_bzip_block_into(first, &mut decoded)?;
    }
    Ok(decoded)
}

const WHOLE_FILE_PAYLOAD: &str = "whole-file compressed Level II payload";
const WHOLE_FILE_METADATA: &str = "whole-file compressed Level II metadata record";

/// Expand a whole-file gzip (every member) or bzip2 wrapper around `raw`,
/// bounded by [`MAX_DECODED_RADAR_BYTES`]; `None` when `raw` has neither.
fn expand_whole_file(raw: &[u8], context: &'static str) -> Result<Option<Vec<u8>>> {
    if raw.starts_with(&[0x1f, 0x8b]) {
        crate::gzip::inflate_gzip_members_limited(raw, MAX_DECODED_RADAR_BYTES, context)
            .map(Some)
            .map_err(NexradError::Compression)
    } else if raw.starts_with(b"BZh") {
        let mut expanded = Vec::new();
        crate::decompress_bzip2_stream_into(raw, &mut expanded, MAX_DECODED_RADAR_BYTES, context)?;
        Ok(Some(expanded))
    } else {
        Ok(None)
    }
}

/// The start of a whole-file wrapper's expansion.
struct WholeFilePrefix {
    bytes: Vec<u8>,
    /// `bytes` is the complete expansion, not just the requested prefix.
    complete: bool,
}

/// The first `prefix_len` bytes of a whole-file wrapper's expansion (or
/// more), or `None` when `raw` has no wrapper. gzip is inflated only that
/// far; bzip2 is decoded block by block until the prefix is covered
/// ([`crate::bzip2_prefix`]), or whole (bounded by
/// [`MAX_DECODED_RADAR_BYTES`]) when the stream cannot be cut into blocks.
fn expand_whole_file_prefix(raw: &[u8], prefix_len: usize) -> Result<Option<WholeFilePrefix>> {
    if raw.starts_with(&[0x1f, 0x8b]) {
        let reader = crate::gzip::MultiGzReader::new(raw).take(prefix_len as u64);
        let bytes = bounded_read::read_to_end_limited(reader, prefix_len, WHOLE_FILE_METADATA)
            .map_err(NexradError::Compression)?;
        Ok(Some(WholeFilePrefix {
            bytes,
            complete: false,
        }))
    } else {
        if raw.starts_with(b"BZh")
            && let Some((bytes, complete)) =
                crate::bzip2_prefix::decode_prefix(raw, prefix_len, WHOLE_FILE_METADATA)?
        {
            return Ok(Some(WholeFilePrefix { bytes, complete }));
        }
        Ok(
            expand_whole_file(raw, WHOLE_FILE_METADATA)?.map(|bytes| WholeFilePrefix {
                bytes,
                complete: true,
            }),
        )
    }
}

/// Length of the big-endian `i32` control word before each LDM record.
const LDM_CONTROL_LEN: usize = 4;

/// True when `records` starts with an LDM control word and a bzip2 stream.
pub(crate) fn starts_with_ldm_record(records: &[u8]) -> bool {
    records.get(LDM_CONTROL_LEN..LDM_CONTROL_LEN + 3) == Some(b"BZh")
}

/// Where the first LDM record of `records` ends (its control word plus the
/// byte count that word gives); `None` when `records` has no control word,
/// or a zero one.
fn first_ldm_record_end(records: &[u8]) -> Option<usize> {
    let control = i32::from_be_bytes(records.get(..LDM_CONTROL_LEN)?.try_into().ok()?);
    if control == 0 {
        return None;
    }
    usize::try_from(control.unsigned_abs())
        .ok()?
        .checked_add(LDM_CONTROL_LEN)
}

/// Split LDM-compressed records: each is a big-endian `i32` byte count
/// (negative for the last record) followed by a bzip2 stream. A zero count,
/// or a lone `-1` in the last four bytes, ends the records. Returns `None`
/// when `records` does not start with that structure.
fn ldm_blocks(records: &[u8]) -> Result<Option<Vec<&[u8]>>> {
    const CONTROL_LEN: usize = LDM_CONTROL_LEN;
    if !starts_with_ldm_record(records) {
        return Ok(None);
    }
    let mut blocks = Vec::new();
    let mut cursor = 0;
    while cursor + CONTROL_LEN <= records.len() {
        let control = i32::from_be_bytes([
            records[cursor],
            records[cursor + 1],
            records[cursor + 2],
            records[cursor + 3],
        ]);
        let end_marker = control == -1 && cursor + CONTROL_LEN == records.len();
        if control == 0 || end_marker {
            break;
        }
        cursor += CONTROL_LEN;
        let len = control.unsigned_abs() as usize;
        let end = cursor.checked_add(len).filter(|end| *end <= records.len());
        let Some(end) = end else {
            return Err(NexradError::Truncated {
                what: "LDM compressed record",
                offset: cursor,
                needed: len,
                available: records.len() - cursor,
            });
        };
        if !records[cursor..end].starts_with(b"BZh") {
            return Err(NexradError::Compression(format!(
                "LDM record at offset {} is not a bzip2 stream",
                cursor - CONTROL_LEN
            )));
        }
        blocks.push(&records[cursor..end]);
        cursor = end;
        if control < 0 {
            break;
        }
    }
    Ok(Some(blocks))
}
