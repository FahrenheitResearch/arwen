//! Digital Radar Data Generic Format (message 31, ICD 2620002AA Table XVII),
//! decoded completely: the Data Header Block, the volume (VOL), elevation
//! (ELV) and radial (RAD) constant blocks, every data moment block including
//! CFP, and blocks with names the ICD does not define.
//!
//! This decoder serves [`MessageWalker`](super::MessageWalker) and metadata
//! extraction. [`crate::read_volume_from_bytes`] keeps its own fast path
//! for the moment grids and reads only the fields it needs.
//!
//! # Layouts across RDA builds
//!
//! Layouts are selected from the sizes the message itself declares, never
//! from an assumed build:
//!
//! | Structure | Size | Selected by | Builds in the corpus |
//! |---|---|---|---|
//! | Data Header Block | 68 bytes (9 pointer slots) | first block pointer | Build 10.0 to 18.2, TDWR |
//! | Data Header Block | 72 bytes (10 slots, CFP added) | first block pointer | Build 19.1 on |
//! | VOL | 44 bytes | LRTUP < 52 | Build 10.0 to 19.1, TDWR |
//! | VOL | 52 bytes (adds ZDR bias estimate) | LRTUP >= 52 | Build 20.1 on |
//! | ELV | 12 bytes | LRTUP | all |
//! | RAD | 20 bytes | LRTUP < 28 | Build 10.0 to 13.2, TDWR |
//! | RAD | 28 bytes (adds calibration constants) | LRTUP >= 28 | Build 14.0 on |
//!
//! ICD note 32 says future builds may append fields to a block, so a block
//! larger than the newest known layout decodes with that layout and the
//! extra bytes ignored. A block smaller than the oldest layout is an error.
//!
//! The data block count is the number of pointers in use, not the number of
//! slots: every radial in the corpus writes its nonzero pointers first, then
//! zeroed slots up to the fixed 68- or 72-byte header (for example 4 pointers
//! and 5 zero slots for a reflectivity-only Build 10 radial). The decoder
//! reads `data_block_count` pointers, which also covers future builds that
//! add blocks (Table XVII-A note 9).
//!
//! # Units
//!
//! Scaled integers keep their raw field (`*_raw`) with the ICD scale noted,
//! and a method returns the physical value. Floating point fields hold the
//! physical value directly; the field name ends in its unit.

use std::borrow::Cow;
use std::fmt;

use chrono::{DateTime, Utc};
use flate2::read::ZlibDecoder;
use recast_radar_core::bounded_read;
use recast_radar_core::model::FieldName;

use crate::RadialStatus;

use super::MessageBody;
use crate::{NexradError, Result};

/// Length of the fixed part of the Data Header Block (bytes 0-31), before
/// the data block pointers.
pub const DATA_HEADER_FIXED_LEN: usize = 32;
/// Length of one data block pointer.
pub const BLOCK_POINTER_LEN: usize = 4;
/// Length of a constant block's type, name and LRTUP (size) fields.
pub const CONSTANT_BLOCK_PREFIX_LEN: usize = 6;
/// VOL block size before Build 20.0.
pub const VOLUME_BLOCK_LEN: usize = 44;
/// VOL block size from Build 20.0 (ZDR bias estimate added).
pub const VOLUME_BLOCK_ZDR_BIAS_LEN: usize = 52;
/// ELV block size.
pub const ELEVATION_BLOCK_LEN: usize = 12;
/// RAD block size before Build 14.0.
pub const RADIAL_BLOCK_LEN: usize = 20;
/// RAD block size from Build 14.0 (calibration constants added).
pub const RADIAL_BLOCK_CALIBRATED_LEN: usize = 28;
/// Length of a data moment block's descriptor, before the gate data.
pub const MOMENT_BLOCK_HEADER_LEN: usize = 28;

/// Table XVII-I typical offset of the "ZDR" data moment, which the VOL
/// block's ZDR bias estimate shares (note 33): `dB = (raw - 418) / 32`. Note
/// 20 says to convert with the scale and offset of the radial's own ZDR
/// moment block, which could change from radial to radial; this is the
/// fallback when the radial has no ZDR block. The encoding has changed
/// between builds: Builds 12 to 18 write 8-bit ZDR with scale 16 and offset
/// 128 (their 44-byte VOL has no estimate), and every volume with the
/// 52-byte VOL layout in the corpus writes these values.
pub const ZDR_OFFSET: f32 = 418.0;
/// Table XVII-I typical scale of the "ZDR" data moment (see [`ZDR_OFFSET`]).
pub const ZDR_SCALE: f32 = 32.0;

/// One decoded message 31 radial (Table XVII).
#[derive(Clone, Debug, PartialEq)]
pub struct DigitalRadarDataGeneric<'a> {
    /// Data Header Block (Table XVII-A).
    pub header: DataHeaderBlock,
    /// Volume Data Constant block (Table XVII-E), when a pointer names one.
    pub volume: Option<VolumeDataBlock>,
    /// Elevation Data Constant block (Table XVII-F).
    pub elevation: Option<ElevationDataBlock>,
    /// Radial Data Constant block (Table XVII-H).
    pub radial: Option<RadialDataBlock>,
    /// Data moment blocks (Table XVII-B) in pointer order, including CFP and
    /// moments with names Table XVII-I does not define.
    pub moments: Vec<MomentDataBlock<'a>>,
    /// Blocks that are neither a known constant block nor a data moment
    /// block, preserved by type and name, in pointer order.
    pub unknown_blocks: Vec<UnknownDataBlock<'a>>,
}

impl<'a> DigitalRadarDataGeneric<'a> {
    /// Decode a message 31 body: the bytes after the 16-byte message header,
    /// starting at the Data Header Block. Block pointers are offsets from the
    /// start of `body`.
    ///
    /// Radials whose compression indicator is BZIP2 or zlib are inflated
    /// from the first block pointer on (the Data Header Block itself is not
    /// compressed, and in every uncompressed radial in the corpus it ends at
    /// the first block pointer), bounded by the header's radial length; their
    /// gate data is then owned. No Archive II file in the corpus has a
    /// compressed radial, so this path follows the ICD without a verified
    /// real sample.
    ///
    /// Errors: a body shorter than its header or pointer table, a pointer
    /// inside the header or past the end, a constant block smaller than its
    /// oldest layout or running past the end, a repeated VOL/ELV/RAD block, a
    /// moment block whose data word size is not a positive multiple of 8 or
    /// whose gates run past the end, and an unknown compression code.
    pub fn decode(body: &'a [u8]) -> Result<Self> {
        let header = DataHeaderBlock::decode(body)?;
        match header.compression {
            CompressionIndicator::Uncompressed => decode_blocks(body, header, true),
            CompressionIndicator::Bzip2 | CompressionIndicator::Zlib => {
                let radial = inflate_radial(body, &header)?;
                decode_blocks(&radial, header, true).map(DigitalRadarDataGeneric::into_owned)
            }
            CompressionIndicator::Unknown(code) => Err(NexradError::InvalidMessage {
                offset: 16,
                reason: format!("message 31 compression indicator {code} is not defined"),
            }),
        }
    }

    /// [`Self::decode`] without the data moment and unknown blocks (left
    /// empty): the Data Header Block and the VOL, ELV and RAD blocks, with
    /// the same checks on them and on every pointer. The metadata reader
    /// takes these from every radial.
    pub(crate) fn decode_constant_blocks(body: &[u8]) -> Result<DigitalRadarDataGeneric<'static>> {
        let header = DataHeaderBlock::decode(body)?;
        match header.compression {
            CompressionIndicator::Uncompressed => {
                decode_blocks(body, header, false).map(DigitalRadarDataGeneric::into_owned)
            }
            _ => {
                let mut radial = DigitalRadarDataGeneric::decode(body)?.into_owned();
                radial.moments.clear();
                radial.unknown_blocks.clear();
                Ok(radial)
            }
        }
    }

    /// Copy borrowed gate data and unknown block bytes so the radial no
    /// longer borrows the input.
    pub fn into_owned(self) -> DigitalRadarDataGeneric<'static> {
        DigitalRadarDataGeneric {
            header: self.header,
            volume: self.volume,
            elevation: self.elevation,
            radial: self.radial,
            moments: self
                .moments
                .into_iter()
                .map(MomentDataBlock::into_owned)
                .collect(),
            unknown_blocks: self
                .unknown_blocks
                .into_iter()
                .map(UnknownDataBlock::into_owned)
                .collect(),
        }
    }

    /// The first data moment block with this name.
    pub fn moment(&self, name: DataMomentName) -> Option<&MomentDataBlock<'a>> {
        self.moments.iter().find(|moment| moment.name == name)
    }

    /// The VOL block's ZDR bias estimate in dB, converted with this radial's
    /// ZDR moment block as Table XVII-E notes 20 and 33 require
    /// ([`VolumeDataBlock::zdr_bias_estimate_db`]). `None` without a VOL
    /// block, in the 44-byte VOL layout, or when the RPG reports the
    /// estimate as not available.
    pub fn zdr_bias_estimate_db(&self) -> Option<f32> {
        self.volume
            .as_ref()?
            .zdr_bias_estimate_db(self.moment(DataMomentName::DifferentialReflectivity))
    }
}

/// Walker hook: the typed body for message 31.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    let radial = match body {
        Cow::Borrowed(bytes) => DigitalRadarDataGeneric::decode(bytes)?,
        Cow::Owned(bytes) => DigitalRadarDataGeneric::decode(&bytes)?.into_owned(),
    };
    Ok(MessageBody::DigitalRadarDataGeneric(Box::new(radial)))
}

fn invalid(offset: usize, reason: String) -> NexradError {
    NexradError::InvalidMessage { offset, reason }
}

/// Inflate the compressed part of a radial and prepend the uncompressed
/// Data Header Block, so pointers index the result.
fn inflate_radial(body: &[u8], header: &DataHeaderBlock) -> Result<Vec<u8>> {
    let header_len = header.blocks_offset();
    let radial_len = usize::from(header.radial_length);
    if radial_len < header_len || body.len() < header_len {
        return Err(invalid(
            18,
            format!(
                "compressed message 31 radial of {} bytes (radial length {radial_len}) does not hold its {header_len}-byte header",
                body.len()
            ),
        ));
    }
    let compressed = &body[header_len..];
    let limit = radial_len - header_len;
    let inflated = match header.compression {
        CompressionIndicator::Bzip2 => {
            let mut inflated = Vec::new();
            crate::decompress_bzip2_stream_into(
                compressed,
                &mut inflated,
                limit,
                "BZIP2-compressed message 31 radial",
            )?;
            inflated
        }
        _ => bounded_read::read_to_end_limited(
            ZlibDecoder::new(compressed),
            limit,
            "zlib-compressed message 31 radial",
        )
        .map_err(NexradError::Compression)?,
    };
    let mut radial = Vec::with_capacity(header_len + inflated.len());
    radial.extend_from_slice(&body[..header_len]);
    radial.extend_from_slice(&inflated);
    Ok(radial)
}

/// The blocks of a radial; data moment and unknown blocks only when
/// `with_moments`.
fn decode_blocks(
    radial: &[u8],
    header: DataHeaderBlock,
    with_moments: bool,
) -> Result<DigitalRadarDataGeneric<'_>> {
    let header_len = header.pointer_table_len();
    let mut decoded = DigitalRadarDataGeneric {
        volume: None,
        elevation: None,
        radial: None,
        moments: Vec::new(),
        unknown_blocks: Vec::new(),
        header,
    };
    for &pointer in &decoded.header.block_pointers {
        if pointer == 0 {
            continue;
        }
        let offset = pointer as usize;
        if offset < header_len || offset > radial.len().saturating_sub(4) {
            return Err(invalid(
                offset,
                format!(
                    "message 31 block pointer {pointer} is outside the blocks ({header_len} to {} bytes)",
                    radial.len()
                ),
            ));
        }
        let block_type = radial[offset];
        let name = [radial[offset + 1], radial[offset + 2], radial[offset + 3]];
        match (block_type, &name) {
            (b'R', b"VOL") => {
                let block = VolumeDataBlock::decode(constant_block(radial, offset, "VOL")?)?;
                set_once(&mut decoded.volume, block, offset, "VOL")?;
            }
            (b'R', b"ELV") => {
                let block = ElevationDataBlock::decode(constant_block(radial, offset, "ELV")?)?;
                set_once(&mut decoded.elevation, block, offset, "ELV")?;
            }
            (b'R', b"RAD") => {
                let block = RadialDataBlock::decode(constant_block(radial, offset, "RAD")?)?;
                set_once(&mut decoded.radial, block, offset, "RAD")?;
            }
            _ if !with_moments => {}
            (b'D', _) => decoded
                .moments
                .push(MomentDataBlock::decode(radial, offset)?),
            _ => {
                let end = unknown_block_end(radial, offset, &decoded.header.block_pointers);
                decoded.unknown_blocks.push(UnknownDataBlock {
                    pointer,
                    block_type,
                    name,
                    bytes: Cow::Borrowed(&radial[offset..end]),
                });
            }
        }
    }
    Ok(decoded)
}

/// The bytes of a constant block, sized by its LRTUP field.
fn constant_block<'a>(radial: &'a [u8], offset: usize, name: &str) -> Result<&'a [u8]> {
    crate::require_len(
        radial,
        offset,
        CONSTANT_BLOCK_PREFIX_LEN,
        "message 31 constant block",
    )?;
    let size = usize::from(crate::be_u16(radial, offset + 4));
    if radial.len() - offset < size {
        return Err(NexradError::Truncated {
            what: "message 31 constant block",
            offset,
            needed: size,
            available: radial.len() - offset,
        });
    }
    let block = &radial[offset..offset + size];
    let oldest = match name {
        "VOL" => VOLUME_BLOCK_LEN,
        "ELV" => ELEVATION_BLOCK_LEN,
        _ => RADIAL_BLOCK_LEN,
    };
    if size < oldest {
        return Err(invalid(
            offset,
            format!(
                "message 31 {name} block declares {size} bytes, less than the {oldest}-byte layout"
            ),
        ));
    }
    Ok(block)
}

fn set_once<T>(slot: &mut Option<T>, block: T, offset: usize, name: &str) -> Result<()> {
    if slot.is_some() {
        return Err(invalid(
            offset,
            format!("message 31 has a second {name} block"),
        ));
    }
    *slot = Some(block);
    Ok(())
}

/// End of an unknown block: its LRTUP size for an `R` block when that fits,
/// otherwise the next higher block pointer, otherwise the end of the radial.
fn unknown_block_end(radial: &[u8], offset: usize, pointers: &[u32]) -> usize {
    let next = pointers
        .iter()
        .map(|&pointer| pointer as usize)
        .filter(|&pointer| pointer > offset && pointer <= radial.len())
        .min()
        .unwrap_or(radial.len());
    if radial[offset] == b'R' && offset + CONSTANT_BLOCK_PREFIX_LEN <= radial.len() {
        let size = usize::from(crate::be_u16(radial, offset + 4));
        if size >= CONSTANT_BLOCK_PREFIX_LEN && offset + size <= radial.len() {
            return offset + size;
        }
    }
    next
}

/// Data Header Block (Table XVII-A).
#[derive(Clone, Debug, PartialEq)]
pub struct DataHeaderBlock {
    /// Bytes 0-3: ICAO radar identifier, as recorded (KVWX 2008-04-15 has
    /// four spaces).
    pub radar_identifier: [u8; 4],
    /// Bytes 4-7: collection time, milliseconds past midnight UTC.
    pub collection_time_ms: u32,
    /// Bytes 8-9: modified Julian date (1 January 1970 is day 1).
    pub modified_julian_date: u16,
    /// Bytes 10-11: radial number within the elevation scan (1 to 720).
    pub azimuth_number: u16,
    /// Bytes 12-15: azimuth angle, degrees.
    pub azimuth_angle_deg: f32,
    /// Byte 16: compression of the blocks after this header.
    pub compression: CompressionIndicator,
    /// Byte 17: spare.
    pub spare: u8,
    /// Bytes 18-19: uncompressed radial length in bytes, this header
    /// included. Some Build 10 radials declare one byte less than the
    /// message body, which is padded to a halfword.
    pub radial_length: u16,
    /// Byte 20: commanded azimuthal spacing.
    pub azimuth_resolution: AzimuthResolution,
    /// Byte 21: radial status (Table III-C); bit 7 marks bad data.
    pub radial_status_code: u8,
    /// Byte 22: elevation number within the volume scan (1 to 32).
    pub elevation_number: u8,
    /// Byte 23: sector number within the cut (0 to 3; 0 only for continuous
    /// surveillance cuts).
    pub cut_sector_number: u8,
    /// Bytes 24-27: elevation angle, degrees.
    pub elevation_angle_deg: f32,
    /// Byte 28: spot blanking status.
    pub spot_blanking: SpotBlankingStatus,
    /// Byte 29: azimuth indexing angle in 0.01 degree steps; 0 means no
    /// indexing.
    pub azimuth_indexing_raw: u8,
    /// Bytes 30-31: number of data block pointers.
    pub data_block_count: u16,
    /// Bytes 32 onward: `data_block_count` data block pointers, offsets from
    /// the start of this header. A zero pointer references no block.
    pub block_pointers: Vec<u32>,
}

impl DataHeaderBlock {
    /// Decode the header and pointer table at the start of a message 31
    /// body.
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, DATA_HEADER_FIXED_LEN, "message 31 data header")?;
        let data_block_count = crate::be_u16(body, 30);
        let header_len = DATA_HEADER_FIXED_LEN + usize::from(data_block_count) * BLOCK_POINTER_LEN;
        crate::require_len(body, 0, header_len, "message 31 data block pointers")?;
        let block_pointers = (0..usize::from(data_block_count))
            .map(|index| crate::be_u32(body, DATA_HEADER_FIXED_LEN + index * BLOCK_POINTER_LEN))
            .collect();
        Ok(Self {
            radar_identifier: [body[0], body[1], body[2], body[3]],
            collection_time_ms: crate::be_u32(body, 4),
            modified_julian_date: crate::be_u16(body, 8),
            azimuth_number: crate::be_u16(body, 10),
            azimuth_angle_deg: crate::be_f32(body, 12),
            compression: CompressionIndicator::from_code(body[16]),
            spare: body[17],
            radial_length: crate::be_u16(body, 18),
            azimuth_resolution: AzimuthResolution::from_code(body[20]),
            radial_status_code: body[21],
            elevation_number: body[22],
            cut_sector_number: body[23],
            elevation_angle_deg: crate::be_f32(body, 24),
            spot_blanking: SpotBlankingStatus(body[28]),
            azimuth_indexing_raw: body[29],
            data_block_count,
            block_pointers,
        })
    }

    /// Bytes read for the header: 32 plus 4 per data block pointer. The
    /// recorded header can be longer, with zeroed pointer slots (see
    /// [`Self::blocks_offset`]).
    pub fn pointer_table_len(&self) -> usize {
        DATA_HEADER_FIXED_LEN + self.block_pointers.len() * BLOCK_POINTER_LEN
    }

    /// Offset of the first block: the lowest nonzero pointer, or
    /// [`Self::pointer_table_len`] when there is none. This is where the Data
    /// Header Block ends, 68 bytes through Build 18.2 and 72 bytes from Build
    /// 19.1 in the corpus.
    pub fn blocks_offset(&self) -> usize {
        self.block_pointers
            .iter()
            .filter(|&&pointer| pointer != 0)
            .min()
            .map_or_else(|| self.pointer_table_len(), |&pointer| pointer as usize)
            .max(self.pointer_table_len())
    }

    /// Radar identifier with trailing spaces and NULs removed ("" when the
    /// identifier is blank).
    pub fn radar_identifier_str(&self) -> String {
        crate::ascii_trim(&self.radar_identifier)
    }

    /// The radar identifier, or `volume_header_icao` (trimmed) when the
    /// identifier is blank, or "" when both are. The walker reads records
    /// without the volume header, so the caller passes its ICAO (bytes 20-23
    /// of the file).
    pub fn radar_identifier_or(&self, volume_header_icao: &str) -> String {
        crate::radar_identifier_or(&self.radar_identifier, volume_header_icao)
    }

    /// Collection time from the modified Julian date and milliseconds.
    pub fn collection_time(&self) -> DateTime<Utc> {
        crate::nexrad_date_ms_to_datetime(
            u32::from(self.modified_julian_date),
            self.collection_time_ms,
        )
    }

    /// Radial status without the bad-data bit.
    pub fn radial_status(&self) -> RadialStatus {
        RadialStatus::from(self.radial_status_code & 0x7F)
    }

    /// True when bit 7 of the radial status (bad data, Table III-C) is set.
    pub fn is_bad_data(&self) -> bool {
        self.radial_status_code & 0x80 != 0
    }

    /// Azimuth indexing angle in degrees (0.01 to 1.00), or `None` when the
    /// azimuth is not keyed to constant angles.
    pub fn azimuth_indexing_deg(&self) -> Option<f32> {
        (self.azimuth_indexing_raw != 0).then(|| f32::from(self.azimuth_indexing_raw) * 0.01)
    }
}

/// Compression indicator (Table XVII-A byte 16).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CompressionIndicator {
    /// 0.
    Uncompressed,
    /// 1: blocks after the header compressed with BZIP2.
    Bzip2,
    /// 2: blocks after the header compressed with zlib.
    Zlib,
    /// 3 (future use) or any other code.
    Unknown(u8),
}

impl CompressionIndicator {
    /// Map a byte 16 code.
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Uncompressed,
            1 => Self::Bzip2,
            2 => Self::Zlib,
            other => Self::Unknown(other),
        }
    }

    /// The byte 16 code.
    pub fn code(self) -> u8 {
        match self {
            Self::Uncompressed => 0,
            Self::Bzip2 => 1,
            Self::Zlib => 2,
            Self::Unknown(code) => code,
        }
    }
}

/// Azimuthal spacing between adjacent radials (Table XVII-A byte 20).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AzimuthResolution {
    /// 1: 0.5 degree.
    HalfDegree,
    /// 2: 1.0 degree.
    OneDegree,
    /// Any other code.
    Unknown(u8),
}

impl AzimuthResolution {
    /// Map a byte 20 code.
    pub fn from_code(code: u8) -> Self {
        match code {
            1 => Self::HalfDegree,
            2 => Self::OneDegree,
            other => Self::Unknown(other),
        }
    }

    /// The byte 20 code.
    pub fn code(self) -> u8 {
        match self {
            Self::HalfDegree => 1,
            Self::OneDegree => 2,
            Self::Unknown(code) => code,
        }
    }

    /// Spacing in degrees, or `None` for an unknown code.
    pub fn degrees(self) -> Option<f32> {
        match self {
            Self::HalfDegree => Some(0.5),
            Self::OneDegree => Some(1.0),
            Self::Unknown(_) => None,
        }
    }
}

/// Radial spot blanking status bits (Table XVII-A byte 28, note 8): 0 when
/// spot blanking is disabled, 4 when enabled with no blanked radials in the
/// cut, 6 when the cut has blanked radials but this one is not, 7 when this
/// radial is blanked.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpotBlankingStatus(pub u8);

impl SpotBlankingStatus {
    /// Bit 0: this radial is spot blanked.
    pub fn radial(self) -> bool {
        self.0 & 0x01 != 0
    }

    /// Bit 1: the elevation scan has spot blanked radials.
    pub fn elevation(self) -> bool {
        self.0 & 0x02 != 0
    }

    /// Bit 2: spot blanking is enabled for the volume scan.
    pub fn volume(self) -> bool {
        self.0 & 0x04 != 0
    }
}

/// Volume Data Constant block, "RVOL" (Table XVII-E).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeDataBlock {
    /// Bytes 4-5 (LRTUP): block size in bytes, which selects the layout.
    pub block_size: u16,
    /// Byte 6: major version; a structural change raises it (1 to Build 13,
    /// 2 from Build 14, 3 from Build 20 in the corpus).
    pub version_major: u8,
    /// Byte 7: minor version; added moment parameters raise it.
    pub version_minor: u8,
    /// Bytes 8-11: site latitude, degrees.
    pub latitude_deg: f32,
    /// Bytes 12-15: site longitude, degrees (east positive).
    pub longitude_deg: f32,
    /// Bytes 16-17: height of the site base above mean sea level, meters.
    pub site_height_m: i16,
    /// Bytes 18-19: height of the feedhorn (the tower, in Build 24 wording)
    /// above ground level, meters.
    pub feedhorn_height_m: u16,
    /// Bytes 20-23: reflectivity calibration constant dBZ0 without the
    /// ground noise scaling of the adaptation data, dB.
    pub calibration_constant_db: f32,
    /// Bytes 24-27: SHV transmitter power, horizontal channel, kW.
    pub horizontal_shv_tx_power_kw: f32,
    /// Bytes 28-31: SHV transmitter power, vertical channel, kW.
    pub vertical_shv_tx_power_kw: f32,
    /// Bytes 32-35: system differential reflectivity calibration, dB.
    pub system_differential_reflectivity_db: f32,
    /// Bytes 36-39: initial system differential phase, degrees.
    pub initial_system_differential_phase_deg: f32,
    /// Bytes 40-41: volume coverage pattern number.
    pub vcp_number: u16,
    /// Bytes 42-43: processing option bits (spare, zero, before Build 14).
    pub processing_status: ProcessingStatus,
    /// Bytes 44-45 of the 52-byte layout (Build 20.0 on): RPG weighted mean
    /// ZDR bias estimate, encoded like the ZDR moment (0 = not available);
    /// `None` in the 44-byte layout. Bytes 46-51 are spare.
    pub zdr_bias_estimate_raw: Option<u16>,
}

impl VolumeDataBlock {
    /// Decode a VOL block (at least [`VOLUME_BLOCK_LEN`] bytes, starting at
    /// its type byte).
    pub fn decode(block: &[u8]) -> Result<Self> {
        crate::require_len(block, 0, VOLUME_BLOCK_LEN, "message 31 VOL block")?;
        Ok(Self {
            block_size: crate::be_u16(block, 4),
            version_major: block[6],
            version_minor: block[7],
            latitude_deg: crate::be_f32(block, 8),
            longitude_deg: crate::be_f32(block, 12),
            site_height_m: crate::be_i16(block, 16),
            feedhorn_height_m: crate::be_u16(block, 18),
            calibration_constant_db: crate::be_f32(block, 20),
            horizontal_shv_tx_power_kw: crate::be_f32(block, 24),
            vertical_shv_tx_power_kw: crate::be_f32(block, 28),
            system_differential_reflectivity_db: crate::be_f32(block, 32),
            initial_system_differential_phase_deg: crate::be_f32(block, 36),
            vcp_number: crate::be_u16(block, 40),
            processing_status: ProcessingStatus(crate::be_u16(block, 42)),
            zdr_bias_estimate_raw: (block.len() >= VOLUME_BLOCK_ZDR_BIAS_LEN)
                .then(|| crate::be_u16(block, 44)),
        })
    }

    /// Layout selected by the block size.
    pub fn layout(&self) -> VolumeBlockLayout {
        if self.zdr_bias_estimate_raw.is_some() {
            VolumeBlockLayout::ZdrBias52
        } else {
            VolumeBlockLayout::Original44
        }
    }

    /// ZDR bias estimate in dB, or `None` in the 44-byte layout or when the
    /// RPG reports it as not available (raw 0).
    ///
    /// Table XVII-E note 33: the estimate is encoded like the "ZDR" data
    /// moment. Note 20: the conversion should use the scale and offset in
    /// the Data Moment Block of the same radial, since they could change
    /// from radial to radial. `zdr` is that radial's ZDR block, which
    /// [`DigitalRadarDataGeneric::zdr_bias_estimate_db`] passes. When the
    /// radial has no ZDR block (Doppler cuts of split cuts carry REF, VEL and
    /// SW only), or the block's scale is 0 (floating-point gates, note 15,
    /// which cannot encode an Integer*2), the Table XVII-I typical values
    /// [`ZDR_OFFSET`] and [`ZDR_SCALE`] are used: `(raw - 418) / 32`.
    pub fn zdr_bias_estimate_db(&self, zdr: Option<&MomentDataBlock<'_>>) -> Option<f32> {
        let raw = self.zdr_bias_estimate_raw.filter(|&raw| raw != 0)?;
        let (scale, offset) = match zdr {
            Some(block) if block.scale != 0.0 => (block.scale, block.offset),
            _ => (ZDR_SCALE, ZDR_OFFSET),
        };
        Some((f32::from(raw) - offset) / scale)
    }
}

/// Layout of a VOL block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VolumeBlockLayout {
    /// 44 bytes, through Build 19.
    Original44,
    /// 52 bytes with the ZDR bias estimate, from Build 20.0.
    ZdrBias52,
}

/// VOL processing status bits (Table XVII-E bytes 42-43, note 28; other bits
/// reserved).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProcessingStatus(pub u16);

impl ProcessingStatus {
    /// Bit 0: receiver noise estimated by RxR noise processing.
    pub fn rxr_noise(self) -> bool {
        self.0 & 0x01 != 0
    }

    /// Bit 1: censoring by CBT (clutter bias threshold) processing.
    pub fn cbt(self) -> bool {
        self.0 & 0x02 != 0
    }
}

/// Elevation Data Constant block, "RELV" (Table XVII-F).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElevationDataBlock {
    /// Bytes 4-5 (LRTUP): block size in bytes.
    pub block_size: u16,
    /// Bytes 6-7: atmospheric attenuation factor, 0.001 dB/km steps.
    pub atmospheric_attenuation_raw: i16,
    /// Bytes 8-11: calibration constant dBZ0 the signal processor used for
    /// this elevation, dB.
    pub calibration_constant_db: f32,
}

impl ElevationDataBlock {
    /// Decode an ELV block (at least [`ELEVATION_BLOCK_LEN`] bytes).
    pub fn decode(block: &[u8]) -> Result<Self> {
        crate::require_len(block, 0, ELEVATION_BLOCK_LEN, "message 31 ELV block")?;
        Ok(Self {
            block_size: crate::be_u16(block, 4),
            atmospheric_attenuation_raw: crate::be_i16(block, 6),
            calibration_constant_db: crate::be_f32(block, 8),
        })
    }

    /// Atmospheric attenuation factor, dB/km (ICD range -0.02 to -0.002).
    pub fn atmospheric_attenuation_db_per_km(&self) -> f32 {
        f32::from(self.atmospheric_attenuation_raw) * 0.001
    }
}

/// Radial Data Constant block, "RRAD" (Table XVII-H).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialDataBlock {
    /// Bytes 4-5 (LRTUP): block size in bytes, which selects the layout.
    pub block_size: u16,
    /// Bytes 6-7: unambiguous range, 0.1 km steps.
    pub unambiguous_range_raw: u16,
    /// Bytes 8-11: noise level, horizontal channel, dBm.
    pub horizontal_noise_level_dbm: f32,
    /// Bytes 12-15: noise level, vertical channel, dBm.
    pub vertical_noise_level_dbm: f32,
    /// Bytes 16-17: Nyquist velocity, 0.01 m/s steps.
    pub nyquist_velocity_raw: u16,
    /// Bytes 18-19: radial flags for RPG processing (set to 0 per ICD;
    /// spare before Build 19).
    pub radial_flags: u16,
    /// Bytes 20-23 of the 28-byte layout (Build 14.0 on): calibration
    /// constant dBZ0, horizontal channel, dBZ.
    pub horizontal_calibration_constant_dbz: Option<f32>,
    /// Bytes 24-27 of the 28-byte layout: calibration constant dBZ0,
    /// vertical channel, dBZ.
    pub vertical_calibration_constant_dbz: Option<f32>,
}

impl RadialDataBlock {
    /// Decode a RAD block (at least [`RADIAL_BLOCK_LEN`] bytes).
    pub fn decode(block: &[u8]) -> Result<Self> {
        crate::require_len(block, 0, RADIAL_BLOCK_LEN, "message 31 RAD block")?;
        let calibrated = block.len() >= RADIAL_BLOCK_CALIBRATED_LEN;
        Ok(Self {
            block_size: crate::be_u16(block, 4),
            unambiguous_range_raw: crate::be_u16(block, 6),
            horizontal_noise_level_dbm: crate::be_f32(block, 8),
            vertical_noise_level_dbm: crate::be_f32(block, 12),
            nyquist_velocity_raw: crate::be_u16(block, 16),
            radial_flags: crate::be_u16(block, 18),
            horizontal_calibration_constant_dbz: calibrated.then(|| crate::be_f32(block, 20)),
            vertical_calibration_constant_dbz: calibrated.then(|| crate::be_f32(block, 24)),
        })
    }

    /// Layout selected by the block size.
    pub fn layout(&self) -> RadialBlockLayout {
        if self.horizontal_calibration_constant_dbz.is_some() {
            RadialBlockLayout::Calibration28
        } else {
            RadialBlockLayout::Original20
        }
    }

    /// Unambiguous range, km.
    pub fn unambiguous_range_km(&self) -> f32 {
        f32::from(self.unambiguous_range_raw) * 0.1
    }

    /// Nyquist velocity, m/s.
    pub fn nyquist_velocity_mps(&self) -> f32 {
        f32::from(self.nyquist_velocity_raw) * 0.01
    }
}

/// Layout of a RAD block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RadialBlockLayout {
    /// 20 bytes, before Build 14.0.
    Original20,
    /// 28 bytes with horizontal and vertical calibration constants, from
    /// Build 14.0.
    Calibration28,
}

/// Name of a data moment block (Table XVII-B bytes 1-3, Table XVII-I).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum DataMomentName {
    /// "REF": reflectivity, dBZ.
    Reflectivity,
    /// "VEL": radial velocity, m/s.
    Velocity,
    /// "SW ": spectrum width, m/s.
    SpectrumWidth,
    /// "ZDR": differential reflectivity, dB.
    DifferentialReflectivity,
    /// "PHI": differential phase, degrees.
    DifferentialPhase,
    /// "RHO": correlation coefficient, unitless.
    CorrelationCoefficient,
    /// "CFP": clutter filter power removed, dB (Build 19 on).
    ClutterFilterPowerRemoved,
    /// A name Table XVII-I does not define, as recorded.
    Other([u8; 3]),
}

impl DataMomentName {
    /// Map the three name bytes. "SW" is accepted with a trailing space or
    /// NUL.
    pub fn from_bytes(name: [u8; 3]) -> Self {
        match &name {
            b"REF" => Self::Reflectivity,
            b"VEL" => Self::Velocity,
            b"SW " | b"SW\0" => Self::SpectrumWidth,
            b"ZDR" => Self::DifferentialReflectivity,
            b"PHI" => Self::DifferentialPhase,
            b"RHO" => Self::CorrelationCoefficient,
            b"CFP" => Self::ClutterFilterPowerRemoved,
            _ => Self::Other(name),
        }
    }

    /// Short name without padding ("SW" for spectrum width).
    pub fn short_name(&self) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            Self::Reflectivity => "REF",
            Self::Velocity => "VEL",
            Self::SpectrumWidth => "SW",
            Self::DifferentialReflectivity => "ZDR",
            Self::DifferentialPhase => "PHI",
            Self::CorrelationCoefficient => "RHO",
            Self::ClutterFilterPowerRemoved => "CFP",
            Self::Other(name) => return Cow::Owned(crate::ascii_trim(name)),
        })
    }

    /// Physical units after scaling, per Table XVII-I ("" for unitless or
    /// unknown moments).
    pub fn units(&self) -> &'static str {
        match self {
            Self::Reflectivity => "dBZ",
            Self::Velocity | Self::SpectrumWidth => "m/s",
            Self::DifferentialReflectivity | Self::ClutterFilterPowerRemoved => "dB",
            Self::DifferentialPhase => "deg",
            Self::CorrelationCoefficient | Self::Other(_) => "",
        }
    }
}

impl fmt::Display for DataMomentName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.short_name())
    }
}

/// Recombination applied to a data moment (Table XVII-B byte 18).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ControlFlags {
    /// 0.
    None,
    /// 1: recombined azimuthal radials.
    RecombinedAzimuthalRadials,
    /// 2: recombined range gates.
    RecombinedRangeGates,
    /// 3: radials and range gates recombined to legacy resolution.
    RecombinedToLegacyResolution,
    /// Any other code.
    Unknown(u8),
}

impl ControlFlags {
    /// Map a byte 18 code.
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => Self::None,
            1 => Self::RecombinedAzimuthalRadials,
            2 => Self::RecombinedRangeGates,
            3 => Self::RecombinedToLegacyResolution,
            other => Self::Unknown(other),
        }
    }

    /// The byte 18 code.
    pub fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::RecombinedAzimuthalRadials => 1,
            Self::RecombinedRangeGates => 2,
            Self::RecombinedToLegacyResolution => 3,
            Self::Unknown(code) => code,
        }
    }
}

/// Data moment block, "D" + name (Table XVII-B), with its gate data.
#[derive(Clone, Debug, PartialEq)]
pub struct MomentDataBlock<'a> {
    /// Bytes 1-3.
    pub name: DataMomentName,
    /// Bytes 4-7: reserved, set to 0.
    pub reserved: u32,
    /// Bytes 8-9: number of gates.
    pub gate_count: u16,
    /// Bytes 10-11: range to the center of the first gate, meters (the ICD
    /// scale is 0.001 km).
    pub first_gate_range_m: u16,
    /// Bytes 12-13: gate spacing, meters (0.001 km).
    pub gate_spacing_m: u16,
    /// Bytes 14-15: TOVER, 0.1 dB steps.
    pub tover_raw: u16,
    /// Bytes 16-17: SNR threshold, 0.125 dB steps (signed; not applied to
    /// CFP).
    pub snr_threshold_raw: i16,
    /// Byte 18.
    pub control_flags: ControlFlags,
    /// Byte 19: bits per gate (8 or 16 in the ICD; always a multiple of 8).
    pub data_word_size: u8,
    /// Bytes 20-23: scale; 0 means floating point gates (note 15).
    pub scale: f32,
    /// Bytes 24-27: offset.
    pub offset: f32,
    /// Gate data from byte 28: `gate_count * data_word_size / 8` bytes,
    /// big-endian words.
    pub data: Cow<'a, [u8]>,
}

impl<'a> MomentDataBlock<'a> {
    /// Decode the data moment block starting at `offset` in `radial`.
    pub fn decode(radial: &'a [u8], offset: usize) -> Result<Self> {
        crate::require_len(
            radial,
            offset,
            MOMENT_BLOCK_HEADER_LEN,
            "message 31 data moment block",
        )?;
        let block = &radial[offset..];
        let gate_count = crate::be_u16(block, 8);
        let data_word_size = block[19];
        if data_word_size == 0 || !data_word_size.is_multiple_of(8) {
            return Err(invalid(
                offset,
                format!(
                    "message 31 data moment word size {data_word_size} is not a positive multiple of 8"
                ),
            ));
        }
        let data_len = usize::from(gate_count) * usize::from(data_word_size / 8);
        crate::require_len(
            radial,
            offset + MOMENT_BLOCK_HEADER_LEN,
            data_len,
            "message 31 data moment gates",
        )?;
        let data_start = offset + MOMENT_BLOCK_HEADER_LEN;
        Ok(Self {
            name: DataMomentName::from_bytes([block[1], block[2], block[3]]),
            reserved: crate::be_u32(block, 4),
            gate_count,
            first_gate_range_m: crate::be_u16(block, 10),
            gate_spacing_m: crate::be_u16(block, 12),
            tover_raw: crate::be_u16(block, 14),
            snr_threshold_raw: crate::be_i16(block, 16),
            control_flags: ControlFlags::from_code(block[18]),
            data_word_size,
            scale: crate::be_f32(block, 20),
            offset: crate::be_f32(block, 24),
            data: Cow::Borrowed(&radial[data_start..data_start + data_len]),
        })
    }

    /// Copy the gate data so the block no longer borrows the input.
    pub fn into_owned(self) -> MomentDataBlock<'static> {
        MomentDataBlock {
            name: self.name,
            reserved: self.reserved,
            gate_count: self.gate_count,
            first_gate_range_m: self.first_gate_range_m,
            gate_spacing_m: self.gate_spacing_m,
            tover_raw: self.tover_raw,
            snr_threshold_raw: self.snr_threshold_raw,
            control_flags: self.control_flags,
            data_word_size: self.data_word_size,
            scale: self.scale,
            offset: self.offset,
            data: Cow::Owned(self.data.into_owned()),
        }
    }

    /// TOVER, dB.
    pub fn tover_db(&self) -> f32 {
        f32::from(self.tover_raw) * 0.1
    }

    /// SNR threshold, dB.
    pub fn snr_threshold_db(&self) -> f32 {
        f32::from(self.snr_threshold_raw) * 0.125
    }

    /// The FM301 field name the volume decoder gives this moment (xradar's
    /// NEXRAD mapping; undefined block names stay verbatim as
    /// [`FieldName::Other`]).
    pub fn field_name(&self) -> FieldName {
        match self.name {
            DataMomentName::Reflectivity => FieldName::Dbzh,
            DataMomentName::Velocity => FieldName::Vradh,
            DataMomentName::SpectrumWidth => FieldName::Wradh,
            DataMomentName::DifferentialReflectivity => FieldName::Zdr,
            DataMomentName::DifferentialPhase => FieldName::Phidp,
            DataMomentName::CorrelationCoefficient => FieldName::Rhohv,
            DataMomentName::ClutterFilterPowerRemoved => FieldName::Ccorh,
            DataMomentName::Other(name) => FieldName::from_nexrad_block(&name),
        }
    }

    /// Gate words as unsigned integers (8, 16 or 32 bits; `None` for other
    /// word sizes).
    pub fn raw_gates(&self) -> Option<impl Iterator<Item = u32> + '_> {
        let width = usize::from(self.data_word_size / 8);
        if !matches!(width, 1 | 2 | 4) {
            return None;
        }
        Some(self.data.chunks_exact(width).map(|word| {
            word.iter()
                .fold(0u32, |value, byte| (value << 8) | u32::from(*byte))
        }))
    }

    /// Classify one gate word. Integer data: codes 0 and 1 are below
    /// threshold and range folded (note 21); for CFP, codes 0 to 7 are
    /// clutter filter states (note 30); other codes convert as
    /// `(raw - offset) / scale`. A zero scale with 32-bit words means IEEE
    /// floating point gates (note 15; no real sample).
    pub fn gate_value(&self, raw: u32) -> GateValue {
        if self.scale == 0.0 {
            return if self.data_word_size == 32 {
                GateValue::Value(f32::from_bits(raw))
            } else {
                GateValue::Undecodable(raw)
            };
        }
        if self.name == DataMomentName::ClutterFilterPowerRemoved {
            match raw {
                0 => return GateValue::ClutterFilterNotApplied,
                1 => return GateValue::PointClutterFilterApplied,
                2 => return GateValue::DualPolOnlyFiltered,
                3..=7 => return GateValue::Reserved(raw),
                _ => {}
            }
        } else {
            match raw {
                0 => return GateValue::BelowThreshold,
                1 => return GateValue::RangeFolded,
                _ => {}
            }
        }
        GateValue::Value(((f64::from(raw) - f64::from(self.offset)) / f64::from(self.scale)) as f32)
    }

    /// Every gate classified by [`Self::gate_value`] (`None` when the word
    /// size is not 8, 16 or 32 bits).
    pub fn gate_values(&self) -> Option<impl Iterator<Item = GateValue> + '_> {
        self.raw_gates()
            .map(|gates| gates.map(|raw| self.gate_value(raw)))
    }
}

/// One classified gate of a data moment.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum GateValue {
    /// Physical value in the moment's units.
    Value(f32),
    /// Code 0 (not CFP): signal below threshold.
    BelowThreshold,
    /// Code 1 (not CFP): range folded.
    RangeFolded,
    /// CFP code 0: clutter filter not applied.
    ClutterFilterNotApplied,
    /// CFP code 1: point clutter filter applied.
    PointClutterFilterApplied,
    /// CFP code 2: dual-pol variables filtered but not single-pol moments.
    DualPolOnlyFiltered,
    /// CFP codes 3 to 7, reserved.
    Reserved(u32),
    /// Zero scale with a word size other than 32 bits.
    Undecodable(u32),
}

/// A block whose type and name this decoder does not know, preserved as
/// bytes (Table XVII-A note 9 asks readers to skip unknown blocks).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnknownDataBlock<'a> {
    /// The pointer that referenced the block.
    pub pointer: u32,
    /// Byte 0: block type character.
    pub block_type: u8,
    /// Bytes 1-3: block name.
    pub name: [u8; 3],
    /// The block from its type byte: an `R` block's LRTUP bytes when that
    /// size fits, otherwise up to the next higher block pointer or the end of
    /// the radial.
    pub bytes: Cow<'a, [u8]>,
}

impl UnknownDataBlock<'_> {
    /// Copy the bytes so the block no longer borrows the input.
    pub fn into_owned(self) -> UnknownDataBlock<'static> {
        UnknownDataBlock {
            pointer: self.pointer,
            block_type: self.block_type,
            name: self.name,
            bytes: Cow::Owned(self.bytes.into_owned()),
        }
    }

    /// Block type and name as text, e.g. "RXYZ".
    pub fn label(&self) -> String {
        let mut label = String::with_capacity(4);
        label.push(char::from(self.block_type));
        label.extend(self.name.iter().map(|byte| char::from(*byte)));
        label
    }
}

/// RDA build number, re-exported from [`super::rda_status`] where the
/// message 2 decoder defines it. Message 31 layouts are selected from block
/// sizes, but the build explains them; see the module documentation.
pub use super::rda_status::RdaBuild;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip() {
        for code in 0..=8u8 {
            assert_eq!(CompressionIndicator::from_code(code).code(), code);
            assert_eq!(AzimuthResolution::from_code(code).code(), code);
            assert_eq!(ControlFlags::from_code(code).code(), code);
        }
        assert_eq!(
            DataMomentName::from_bytes(*b"SW "),
            DataMomentName::SpectrumWidth
        );
        assert_eq!(DataMomentName::from_bytes(*b"KDP").short_name(), "KDP");
    }
}
