//! RDA Log Data (message 33, ICD 2620002AA Table XVIV).
//!
//! Log text the RDA sends to the RPG for field support; it is neither radial
//! data nor metadata and is not recorded in Archive II metadata. No file in
//! the test corpus holds one, so this decoder follows the ICD without a
//! verified real sample.
//!
//! Table XVIV numbers halfwords from 0: version (0-1), identifier (2-14),
//! data version (15-16), compression type (17-18), compressed size (19-20),
//! decompressed size (21-22), spare (23-33), data (34 onward). The appended
//! data is inflated with pure-Rust decoders: gzip via `flate2` (zlib-rs),
//! bzip2 via `recast-radar-bzip2`, and the first member of a ZIP archive
//! (stored or deflate).

use std::borrow::Cow;

use flate2::read::{DeflateDecoder, GzDecoder};

use super::MessageBody;
use crate::{NexradError, Result};

/// Byte offset of the appended data within the message body.
pub const RDA_LOG_DATA_OFFSET: usize = 68;

/// Ceiling on inflated log data. Table XVIV allows sizes up to 2 000 000 000
/// bytes; log messages are text, so a far smaller bound rejects compression
/// bombs.
pub const MAX_RDA_LOG_BYTES: usize = 64 * 1024 * 1024;

const ZIP_LOCAL_HEADER_LEN: usize = 30;

/// Decoded RDA log data message (Table XVIV).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RdaLogData {
    /// Version of the message 33 format.
    pub version: u32,
    /// Log file name, e.g. `AzServoLog`, with NUL padding removed.
    pub identifier: String,
    /// Version of the log identified by `identifier`.
    pub data_version: u32,
    /// Compression of the appended data.
    pub compression: RdaLogCompression,
    /// Declared size of the appended (possibly compressed) data in bytes.
    pub compressed_size: u32,
    /// Declared size of the data after decompression in bytes.
    pub decompressed_size: u32,
    /// The log data, decompressed.
    pub data: Vec<u8>,
}

impl RdaLogData {
    /// Decode a message body (the bytes after the 16-byte message header),
    /// inflating compressed log data up to [`MAX_RDA_LOG_BYTES`].
    pub fn decode(body: &[u8]) -> Result<Self> {
        Self::decode_limited(body, MAX_RDA_LOG_BYTES)
    }

    /// [`Self::decode`] with the log data limited to `limit` bytes (and
    /// never more than [`MAX_RDA_LOG_BYTES`]): data that would be longer,
    /// stored or inflated, is an error. The volume decoder passes what is
    /// left of its RDA log allowance.
    pub fn decode_limited(body: &[u8], limit: usize) -> Result<Self> {
        let limit = limit.min(MAX_RDA_LOG_BYTES);
        crate::require_len(body, 0, RDA_LOG_DATA_OFFSET, "RDA log data header")?;
        let compression = RdaLogCompression::from_code(crate::be_u32(body, 34));
        if let RdaLogCompression::Unknown(code) = compression {
            return Err(NexradError::InvalidMessage {
                offset: 34,
                reason: format!("reserved RDA log compression type {code}"),
            });
        }
        let compressed_size = crate::be_u32(body, 38);
        let decompressed_size = crate::be_u32(body, 42);
        let appended_len = compressed_size as usize;
        crate::require_len(body, RDA_LOG_DATA_OFFSET, appended_len, "RDA log data")?;
        let appended = &body[RDA_LOG_DATA_OFFSET..RDA_LOG_DATA_OFFSET + appended_len];
        let data = match compression {
            RdaLogCompression::Gzip => inflate(GzDecoder::new(appended), "gzip", limit)?,
            RdaLogCompression::Bzip2 => inflate_bzip2(appended, limit)?,
            RdaLogCompression::Zip => inflate_zip_first_member(appended, limit)?,
            RdaLogCompression::Uncompressed | RdaLogCompression::Unknown(_) => {
                if appended.len() > limit {
                    return Err(NexradError::Compression(format!(
                        "uncompressed RDA log data of {} bytes exceeds the {limit}-byte limit",
                        appended.len()
                    )));
                }
                appended.to_vec()
            }
        };
        Ok(Self {
            version: crate::be_u32(body, 0),
            identifier: crate::ascii_trim(&body[4..30]),
            data_version: crate::be_u32(body, 30),
            compression,
            compressed_size,
            decompressed_size,
            data,
        })
    }

    /// The log text with trailing NUL padding removed and invalid UTF-8
    /// replaced.
    pub fn text(&self) -> Cow<'_, str> {
        let end = self
            .data
            .iter()
            .rposition(|byte| *byte != 0)
            .map_or(0, |last| last + 1);
        String::from_utf8_lossy(&self.data[..end])
    }
}

/// Compression type codes (Table XVIV, halfwords 17-18).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum RdaLogCompression {
    /// 0.
    #[default]
    Uncompressed,
    /// 1.
    Gzip,
    /// 2.
    Bzip2,
    /// 3.
    Zip,
    /// Reserved higher values.
    Unknown(u32),
}

impl RdaLogCompression {
    /// Map a Table XVIV code.
    pub fn from_code(code: u32) -> Self {
        match code {
            0 => Self::Uncompressed,
            1 => Self::Gzip,
            2 => Self::Bzip2,
            3 => Self::Zip,
            other => Self::Unknown(other),
        }
    }
}

fn inflate(reader: impl std::io::Read, format: &str, limit: usize) -> Result<Vec<u8>> {
    recast_radar_core::bounded_read::read_to_end_limited(reader, limit, "RDA log data")
        .map_err(|message| NexradError::Compression(format!("{format}: {message}")))
}

fn inflate_bzip2(compressed: &[u8], limit: usize) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    crate::decompress_bzip2_stream_into(compressed, &mut data, limit, "RDA log data")?;
    Ok(data)
}

/// Inflate the first member of a ZIP archive from its local file header.
fn inflate_zip_first_member(bytes: &[u8], limit: usize) -> Result<Vec<u8>> {
    let zip_error = |reason: String| NexradError::Compression(format!("zip: {reason}"));
    if bytes.len() < ZIP_LOCAL_HEADER_LEN || !bytes.starts_with(b"PK\x03\x04") {
        return Err(zip_error(
            "RDA log data is not a ZIP local file record".to_owned(),
        ));
    }
    let le_u16 = |offset: usize| u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
    let le_u32 = |offset: usize| {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    };
    let flags = le_u16(6);
    if flags & 0x0001 != 0 {
        return Err(zip_error("encrypted member".to_owned()));
    }
    let has_data_descriptor = flags & 0x0008 != 0;
    let method = le_u16(8);
    let compressed_size = le_u32(18) as usize;
    let data_start = ZIP_LOCAL_HEADER_LEN + usize::from(le_u16(26)) + usize::from(le_u16(28));
    if data_start > bytes.len() {
        return Err(zip_error(format!(
            "member data starts at {data_start}, past the {}-byte record",
            bytes.len()
        )));
    }
    let data = if has_data_descriptor {
        &bytes[data_start..]
    } else {
        let data_end = data_start
            .checked_add(compressed_size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| zip_error(format!("member of {compressed_size} bytes is truncated")))?;
        &bytes[data_start..data_end]
    };
    match method {
        0 if !has_data_descriptor => {
            if data.len() > limit {
                return Err(zip_error(format!("stored member exceeds {limit} bytes")));
            }
            Ok(data.to_vec())
        }
        0 => Err(zip_error(
            "stored member with a trailing data descriptor has no size".to_owned(),
        )),
        8 => inflate(DeflateDecoder::new(data), "zip deflate", limit),
        other => Err(zip_error(format!("unsupported compression method {other}"))),
    }
}

/// Walker hook: the typed body for message 33.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    RdaLogData::decode(&body).map(MessageBody::RdaLog)
}
