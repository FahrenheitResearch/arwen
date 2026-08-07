//! NEXRAD Level-II (Archive II) file parser.
//! Specification: ICD 2620010H (RDA/RPG).

use byteorder::{BigEndian, ReadBytesExt};
use bzip2::read::BzDecoder;
use chrono::{Datelike, NaiveDate};
use rayon::prelude::*;
use std::io::{Cursor, Read};

use crate::products::RadarProduct;

const VOLUME_HEADER_SIZE: usize = 24;
const MSG_HEADER_SIZE: usize = 16;
/// Message-31 data header block: radar id through data block count, before
/// the pointer list.  4+4+2+2+4+1+1+2+1+1+1+1+4+1+1+2.
const MSG31_HEADER_SIZE: usize = 32;
/// Generic data block header: one type character and a three-character name.
const GENERIC_BLOCK_NAME_SIZE: usize = 4;
/// Moment block header, up to the first gate.
const MOMENT_HEADER_SIZE: usize = 28;

/// Message-31 VOL block layouts this decoder has read, as
/// `(version major, version minor, LRTUP)`.
///
/// Established the same way the RAD layout was — from the bytes of real
/// archived volumes, not from a reading of the ICD.  Fifteen volumes off
/// `unidata-nexrad-level2`, seven sites, 2011 through 2026:
///
/// ```text
/// 1.0 / LRTUP 44  KTLX 2011-05-24, 2012-04-14, 2013-05-20, 2014-04-27
/// 2.0 / LRTUP 44  KTLX 2015-05-06, 2016-05-09, 2017-05-16, 2018-05-01,
///                 2019-05-20, 2020-05-22, 2021-05-26
/// 3.0 / LRTUP 52  KGWX 2022-03-30, KDDC 2024-05-06, KLZK 2025-03-14,
///                 KFWS 2026-07-29, KTLX 2026-07-29
/// ```
///
/// **The 1.0 row is a correction, and the way it was missed is the lesson.**
/// The first survey behind this table sampled 2017 onward, found only 2.0
/// and 3.0, and concluded that the 1.0 the crate's synthetic fixture
/// declared was invented.  It is not: 1.0 is the 2011-2014 era, and it was
/// outside the sampling window rather than outside the archive.  Extending
/// the window to the years the pre-2016 `.gz` keys cover produced it
/// immediately.  A survey is evidence only over the range it actually
/// covers, and the range has to be the range the product will read.
///
/// All three versions put latitude, longitude, site height, VCP and
/// processing status at the same offsets; 1.0 and 2.0 differ in the RAD
/// block beside them (LRTUP 20 against 28, both already accepted) rather
/// than in this one.  Version 3.0 appends eight bytes to the 44-byte layout
/// and moves nothing.
///
/// The pairing is the contract, not either half alone: a block declaring
/// 3.0 in 44 bytes, or 2.0 in 52, is stating two incompatible things about
/// where its fields are, and this decoder reads the first 44 bytes of it.
///
/// A combination outside this table is a layout nothing here has read.  The
/// bytes at 8..16 would still decode to *some* pair of floats, and calling
/// that pair a radar position is the same class of error as the byte-26
/// Nyquist this crate carried until 2026-07-30.  Strict mode refuses it;
/// lenient mode, the renderer's, is unchanged.
const KNOWN_VOL_LAYOUTS: [(u8, u8, usize); 3] = [(1, 0, 44), (2, 0, 44), (3, 0, 52)];

/// Bytes of the VOL block whose field positions this decoder relies on:
/// through the volume coverage pattern and processing status at 40..44.
/// Shared by both layouts in [`KNOWN_VOL_LAYOUTS`].
const VOL_READ_BYTES: usize = 44;

/// The ELV block's only layout: type + name + LRTUP + atmospheric
/// attenuation (int*2) + calibration constant (real*4).  All nine volumes
/// surveyed above declare exactly this, and the arithmetic leaves no room
/// for another arrangement.
const ELV_LRTUP: usize = 12;

/// How much a malformed volume is allowed to get away with.
///
/// The renderer and the observation front door want opposite things from
/// the same bytes.  A renderer that draws 95% of a damaged volume beats one
/// that draws nothing, so [`ParseMode::Lenient`] keeps whatever decodes and
/// stops at the first thing that does not.  An observation that is
/// assimilated cannot be un-assimilated, so [`ParseMode::Strict`] refuses
/// the whole volume, by name, the moment any part of it contradicts the
/// contract it declares about itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseMode {
    /// Decode what can be decoded; a failure ends the walk.
    Lenient,
    /// Any contradiction is a named refusal of the entire volume.
    Strict,
}

impl ParseMode {
    fn is_strict(self) -> bool {
        matches!(self, ParseMode::Strict)
    }
}

/// What one pass of the message walk produced.
enum MessageOutcome {
    Radial(u8, u8, Option<VolSite>, RadialData),
    /// A message this decoder does not read, stepped over.
    Skipped,
    /// No further whole message fits in what remains.
    EndOfData,
}

/// The radar's own statement of where it is, from the Message-31 VOL block.
///
/// This is the self-describing route to the antenna position: every
/// Message-31 radial points at a VOL block carrying the site latitude,
/// longitude, the ground elevation of the site (`site_height_m`, metres
/// MSL) and the height of the feedhorn above it (`feedhorn_height_m`,
/// metres AGL).  The beam origin — the number every gate height is
/// computed from — is `site_height_m + feedhorn_height_m`.  The vendored
/// site table stores neither reliably (130 of 141 elevations are a 0.0
/// placeholder, and the populated ones are ground elevation without the
/// feedhorn), so a consumer that has the volume in hand should prefer
/// this block to any table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolSite {
    pub latitude_deg: f32,
    pub longitude_deg: f32,
    pub site_height_m: i16,
    pub feedhorn_height_m: u16,
}

impl VolSite {
    /// Antenna (beam-origin) height above mean sea level, metres.
    pub fn antenna_height_m(&self) -> f64 {
        f64::from(self.site_height_m) + f64::from(self.feedhorn_height_m)
    }
}

#[derive(Debug, Clone)]
pub struct Level2File {
    pub station_id: String,
    pub volume_date: u16,
    pub volume_time: u32,
    /// First VOL block's site statement, when any radial carried one.
    /// Strict parsing refuses a volume whose radials disagree about it.
    pub vol_site: Option<VolSite>,
    pub sweeps: Vec<Level2Sweep>,
}

#[derive(Debug, Clone)]
pub struct Level2Sweep {
    pub elevation_number: u8,
    pub elevation_angle: f32,
    /// The **smallest** Nyquist velocity any radial in this cut reported.
    ///
    /// One scalar for a whole sweep is a reduction, and the first radial's
    /// value used to be it.  A cut whose first radial says 32 m/s and whose
    /// later radials say 20 m/s would then license a reported 18 m/s gate
    /// under a 25.6 m/s threshold, when that gate's own radial puts the
    /// threshold at 16 m/s.  The minimum is the only reduction that cannot
    /// license a gate its own radial would have rejected.
    pub nyquist_velocity: Option<f32>,
    /// True when the cut's radials did not all report the same Nyquist.
    pub nyquist_radials_disagree: bool,
    /// Sequential sweep index within the volume (0-based).
    pub sweep_index: u16,
    /// Radial status of the first radial (0=start elev, 3=start volume, 5=start elev mid-vol).
    pub start_status: u8,
    /// Radial status of the last radial (2=end elev, 4=end volume; other values indicate incomplete cut).
    pub end_status: u8,
    /// Cut sector number from the VCP (0 = full 360°).
    pub cut_sector: u8,
    pub radials: Vec<RadialData>,
}

#[derive(Debug, Clone)]
pub struct RadialData {
    pub azimuth: f32,
    pub elevation: f32,
    pub azimuth_spacing: f32,
    pub nyquist_velocity: Option<f32>,
    /// Radial status: 0=start elev, 1=intermediate, 2=end elev, 3=start volume,
    /// 4=end volume, 5=start elev (found mid-volume in some SAILS data).
    pub radial_status: u8,
    pub moments: Vec<MomentData>,
}

/// Why a gate in [`MomentData::data`] is not a number.
///
/// Message-31 reserves the two smallest data words of every moment for
/// meanings that are not measurements, and they are **opposites**:
///
/// * raw `0` -- *below threshold*.  The radar illuminated this gate and the
///   return did not clear the significant-return threshold.  That is a
///   detection of nothing, which is a real observation of clear air.
/// * raw `1` -- *range folded*.  The gate's return is ambiguous between
///   trips, so the RDA cannot say where it came from.  It may well be a
///   storm, and it is never evidence of anything.
///
/// `data` maps both to `f32::NAN`, because that is the value every existing
/// consumer of this crate was written against and changing it would move
/// numbers under products that are already published.  The distinction is
/// carried beside it in [`MomentData::censor`] instead, so a consumer that
/// needs it can have it and one that does not is untouched.
///
/// The codes are `u8` and deliberately not an `enum`: they are transcribed
/// verbatim into a `|u1` plane in the observation pack, and a numeric
/// contract that crosses a file boundary should be stated as numbers.
pub mod censor {
    /// `data` holds a decoded measurement.
    pub const MEASURED: u8 = 0;
    /// Raw `0`: the radar looked and detected nothing.  Clear air.
    pub const BELOW_THRESHOLD: u8 = 1;
    /// Raw `1`: second-trip ambiguity.  Never usable as clear air.
    pub const RANGE_FOLDED: u8 = 2;
    /// Not produced by this decoder.  Reserved for a consumer that widens a
    /// moment into a rectangle the radar never filled -- a radial that
    /// carried no such moment at all -- so that "not collected" stays
    /// distinct from "collected, and empty".
    pub const NOT_COLLECTED: u8 = 3;
}

#[derive(Debug, Clone)]
pub struct MomentData {
    pub product: RadarProduct,
    pub gate_count: u16,
    pub first_gate_range: u16,
    pub gate_size: u16,
    pub data: Vec<f32>,
    /// One [`censor`] code per gate, parallel to `data` and always the same
    /// length.  Every `NAN` in `data` has a reason here; every number in
    /// `data` is [`censor::MEASURED`].
    pub censor: Vec<u8>,
}

struct VolumeHeader {
    station_id: String,
    volume_date: u16,
    volume_time: u32,
}

struct MessageHeader {
    message_size: u16,
    message_type: u8,
}

struct Message31Header {
    azimuth_angle: f32,
    elevation_angle: f32,
    elevation_number: u8,
    azimuth_resolution: u8,
    radial_status: u8,
    cut_sector: u8,
    data_block_count: u16,
    compression: u8,
    radial_length: u16,
}

impl Level2File {
    /// Decode a volume, keeping whatever decodes.  The renderer's contract.
    pub fn parse(raw_data: &[u8]) -> Result<Self, String> {
        Self::parse_with(raw_data, ParseMode::Lenient)
    }

    /// Decode a volume, refusing it by name if any part contradicts itself.
    ///
    /// This is the contract an observation front door needs: a volume that
    /// frames cleanly at the LDM layer can still carry a radial whose block
    /// pointer leaves its own message, and following that pointer reads
    /// plausible numbers out of the *next* radial.  Outer framing cannot
    /// see that; only the per-radial envelope can.
    pub fn parse_strict(raw_data: &[u8]) -> Result<Self, String> {
        Self::parse_with(raw_data, ParseMode::Strict)
    }

    pub fn parse_with(raw_data: &[u8], mode: ParseMode) -> Result<Self, String> {
        let header_str = String::from_utf8_lossy(&raw_data[..raw_data.len().min(9)]);

        let data = if header_str.starts_with("AR2V") || header_str.starts_with("ARCH") {
            Self::decompress_archive2(raw_data, mode)?
        } else {
            raw_data.to_vec()
        };

        let mut cursor = Cursor::new(&data);
        let header = Self::read_volume_header(&mut cursor)?;

        // Collect all radials, then split into sweeps by cut boundaries.
        // Uses radial_status (0=start elev, 3=start volume, 5=start elev mid-vol)
        // and elevation_number changes as fallback to properly separate
        // SAILS/MESO-SAILS duplicate elevations into distinct sweeps.
        let mut all_radials: Vec<(u8, u8, RadialData)> = Vec::new();
        let mut vol_site: Option<VolSite> = None;

        while (cursor.position() as usize) < data.len().saturating_sub(MSG_HEADER_SIZE) {
            match Self::read_message(&mut cursor, &data, mode) {
                Ok(MessageOutcome::Radial(elev_num, cut_sector, radial_vol, radial)) => {
                    match (vol_site, radial_vol) {
                        (None, Some(site)) => vol_site = Some(site),
                        (Some(first), Some(site)) if site != first && mode.is_strict() => {
                            return Err(format!(
                                "corrupt Level-II volume: two Message-31 radials disagree \
                                 about the radar's own position — one VOL block says \
                                 {first:?}, another says {site:?}. One antenna cannot be in \
                                 two places; the volume is not one radar's volume"
                            ));
                        }
                        _ => {}
                    }
                    all_radials.push((elev_num, cut_sector, radial));
                }
                Ok(MessageOutcome::Skipped) => continue,
                Ok(MessageOutcome::EndOfData) => break,
                Err(error) => {
                    if mode.is_strict() {
                        return Err(error);
                    }
                    break;
                }
            }
        }

        // A volume that framed cleanly and yielded nothing is not an empty
        // volume, it is a volume this decoder failed to read: the message
        // walk found no Message-31 anywhere.  The renderer can draw zero
        // sweeps, but an observation front door that publishes one has
        // published nothing and called it READY.
        if mode.is_strict() && all_radials.is_empty() {
            return Err(format!(
                "corrupt Level-II volume: {} bytes of message stream carried no Message-31 \
                 radial. Either the framing was misread or the volume holds no moment data; \
                 neither is an observation set",
                data.len()
            ));
        }

        let sweeps = Self::split_radials_into_sweeps(all_radials);

        Ok(Level2File {
            station_id: header.station_id,
            volume_date: header.volume_date,
            volume_time: header.volume_time,
            vol_site,
            sweeps,
        })
    }

    pub fn timestamp_string(&self) -> String {
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let date = epoch + chrono::Duration::days((self.volume_date as i64) - 1);
        let total_secs = self.volume_time / 1000;
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
            date.year(),
            date.month(),
            date.day(),
            hours,
            minutes,
            seconds,
        )
    }

    /// Available products across all sweeps.
    pub fn available_products(&self) -> Vec<RadarProduct> {
        let mut products = std::collections::HashSet::new();
        for sweep in &self.sweeps {
            for radial in &sweep.radials {
                for moment in &radial.moments {
                    products.insert(moment.product);
                }
            }
        }
        let mut list: Vec<RadarProduct> = products.into_iter().collect();
        list.sort_by_key(|p| p.short_name().to_string());
        list
    }

    /// True when the bytes after the volume header are an LDM block table.
    ///
    /// Two shapes carry an Archive-II message stream and the magic does not
    /// tell them apart — it is `AR2V` in both:
    ///
    /// * **LDM + bzip2**, every plain `_V06` key on the open-data buckets:
    ///   a four-byte block length, then a bzip2 stream, repeated to EOF.
    /// * **an uncompressed message stream**, which is what the pre-2016
    ///   `.gz` keys hold once gunzipped: the messages follow the 24-byte
    ///   volume header directly and there is no block table at all.
    ///
    /// The discriminator is the first block's payload.  In an LDM-framed
    /// volume the first block is the metadata block and is always bzip2, so
    /// it begins `BZh`; in the unframed shape those same bytes are the
    /// first message's CTM header.  Checked against real volumes from 2011,
    /// 2012, 2013, 2014, 2015 and 2016 (all unframed, all gzipped in the
    /// archive) and 2017, 2019 and 2026 (all LDM+bzip2).
    ///
    /// A damaged LDM volume whose first block is not bzip2 classifies as
    /// unframed here and then fails the message walk, so it is refused
    /// either way rather than silently half-read.
    ///
    /// Public so the framing check in `rw-nexrad` can be pinned against it:
    /// the framing record and the decoder must agree about which shape they
    /// read, or the record describes a volume the decoder did not parse.
    #[doc(hidden)]
    pub fn is_ldm_framed(raw: &[u8]) -> bool {
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

    fn decompress_archive2(raw_data: &[u8], mode: ParseMode) -> Result<Vec<u8>, String> {
        if raw_data.len() < VOLUME_HEADER_SIZE {
            return Err("Data too short for volume header".into());
        }

        // The pre-2016 archive: no block table, so there is nothing to
        // decompress and the file already *is* the message stream.  The
        // walk in `parse_with` and every strict check inside `read_message`
        // then run over it unchanged -- this branch chooses a framing, it
        // does not relax a contract.
        if !Self::is_ldm_framed(raw_data) {
            return Ok(raw_data.to_vec());
        }

        let mut blocks: Vec<(usize, usize, bool)> = Vec::new();
        let mut pos = VOLUME_HEADER_SIZE;

        while pos + 4 <= raw_data.len() {
            let block_size = i32::from_be_bytes([
                raw_data[pos],
                raw_data[pos + 1],
                raw_data[pos + 2],
                raw_data[pos + 3],
            ]);
            pos += 4;
            let actual_size = block_size.unsigned_abs() as usize;
            if pos + actual_size > raw_data.len() {
                break;
            }
            let is_bz2 = actual_size >= 2 && raw_data[pos] == b'B' && raw_data[pos + 1] == b'Z';
            blocks.push((pos, actual_size, is_bz2));
            pos += actual_size;
        }

        // A bzip2 block that will not decode used to be substituted whole,
        // still compressed, into the message stream.  Compressed bytes
        // parsed as Message-31 are noise that decodes to plausible-looking
        // radials, so the substitution manufactured observations out of a
        // decode failure.  A block that will not decode now contributes
        // nothing (lenient) or refuses the volume (strict); neither invents
        // radials.
        let decompressed: Vec<Result<Vec<u8>, String>> = blocks
            .par_iter()
            .map(|&(start, len, is_bz2)| {
                let block_data = &raw_data[start..start + len];
                if is_bz2 {
                    let mut decoder = BzDecoder::new(block_data);
                    let mut out = Vec::new();
                    match decoder.read_to_end(&mut out) {
                        Ok(_) => Ok(out),
                        Err(error) => Err(format!(
                            "corrupt Level-II volume: the bzip2 LDM block at offset {start} \
                             ({len} bytes) will not decompress: {error}"
                        )),
                    }
                } else {
                    Ok(block_data.to_vec())
                }
            })
            .collect();

        let mut decompressed_ok: Vec<Vec<u8>> = Vec::with_capacity(decompressed.len());
        for block in decompressed {
            match block {
                Ok(bytes) => decompressed_ok.push(bytes),
                Err(error) => {
                    if mode.is_strict() {
                        return Err(error);
                    }
                }
            }
        }
        let decompressed = decompressed_ok;

        let total: usize = VOLUME_HEADER_SIZE + decompressed.iter().map(|b| b.len()).sum::<usize>();
        let mut result = Vec::with_capacity(total);
        result.extend_from_slice(&raw_data[..VOLUME_HEADER_SIZE]);
        for block in decompressed {
            result.extend_from_slice(&block);
        }
        Ok(result)
    }

    fn read_volume_header(cursor: &mut Cursor<&Vec<u8>>) -> Result<VolumeHeader, String> {
        let mut header = [0u8; 24];
        cursor.read_exact(&mut header).map_err(|e| e.to_string())?;

        let filename_str = String::from_utf8_lossy(&header[..12]);
        let icao = String::from_utf8_lossy(&header[20..24]).trim().to_string();

        let station_id = if icao.len() == 4 && icao.chars().all(|c| c.is_ascii_alphanumeric()) {
            icao
        } else {
            filename_str.chars().skip(4).take(4).collect::<String>()
        };

        let volume_date = u16::from_be_bytes([header[14], header[15]]);
        let volume_time = u32::from_be_bytes([header[16], header[17], header[18], header[19]]);

        Ok(VolumeHeader {
            station_id,
            volume_date,
            volume_time,
        })
    }

    /// Read one message, bounding every read to that message's own radial.
    ///
    /// The bound is the point.  Block pointers used to be checked only
    /// against the end of the decompressed volume, so a missing, reordered,
    /// or corrupt pointer read bytes belonging to a *later* radial and got
    /// a plausible answer: geometry, gates, or — because every block whose
    /// first byte was `R` was treated as a Nyquist source — a number from a
    /// VOL or ELV block that could land inside the 4–100 m/s plausibility
    /// band and license velocities nothing supports.  Two independent
    /// declarations bound the radial (the message header's size and the
    /// Message-31 header's own radial length); neither is required to agree
    /// with the other, and the tighter of the two that fits inside the
    /// volume becomes the envelope.
    fn read_message(
        cursor: &mut Cursor<&Vec<u8>>,
        data: &[u8],
        mode: ParseMode,
    ) -> Result<MessageOutcome, String> {
        let strict = mode.is_strict();
        let start_pos = cursor.position() as usize;
        if start_pos + 12 > data.len() {
            return Ok(MessageOutcome::EndOfData);
        }

        let mut ctm = [0u8; 12];
        cursor.read_exact(&mut ctm).map_err(|e| e.to_string())?;

        if (cursor.position() as usize) + MSG_HEADER_SIZE > data.len() {
            return Ok(MessageOutcome::EndOfData);
        }

        let msg_header = Self::read_message_header(cursor)?;

        if msg_header.message_type != 31 {
            let next_pos = start_pos + 2432;
            if next_pos <= data.len() {
                cursor.set_position(next_pos as u64);
            } else {
                return Ok(MessageOutcome::EndOfData);
            }
            return Ok(MessageOutcome::Skipped);
        }

        let msg31_start = cursor.position() as usize;
        let msg31 = Self::read_msg31_header(cursor)?;

        if msg31.compression != 0 && strict {
            return Err(format!(
                "corrupt Level-II volume: the Message-31 radial at offset {start_pos} declares \
                 compression indicator {}; this decoder reads uncompressed radials (0) only, \
                 and decoding a compressed radial as though it were plain would place every \
                 gate at the wrong range",
                msg31.compression
            ));
        }

        let mut block_pointers = Vec::new();
        for _ in 0..msg31.data_block_count {
            let offset = cursor.read_u32::<BigEndian>().map_err(|e| e.to_string())?;
            block_pointers.push(offset);
        }

        let radial_header_bytes = MSG31_HEADER_SIZE + 4 * msg31.data_block_count as usize;
        let envelope_end = Self::radial_envelope_end(
            data,
            start_pos,
            msg31_start,
            &msg_header,
            &msg31,
            radial_header_bytes,
            strict,
        )?;

        let mut moments = Vec::new();
        let mut nyquist_velocity: Option<f32> = None;
        let mut vol_site: Option<VolSite> = None;
        let (mut saw_vol, mut saw_elv, mut saw_rad) = (false, false, false);

        for ptr_offset in &block_pointers {
            // A zero pointer is how the RDA says "this moment is absent".
            if *ptr_offset == 0 {
                continue;
            }
            let offset = *ptr_offset as usize;
            let block_pos = msg31_start + offset;
            if offset < radial_header_bytes
                || block_pos + GENERIC_BLOCK_NAME_SIZE > envelope_end
            {
                if strict {
                    return Err(format!(
                        "corrupt Level-II volume: the Message-31 radial at offset {start_pos} \
                         points a data block at {offset} bytes, outside its own \
                         {}-byte radial (header is {radial_header_bytes} bytes). Following it \
                         would read a neighbouring radial's bytes as this one's",
                        envelope_end - msg31_start
                    ));
                }
                continue;
            }

            let block_type = data[block_pos];
            let name = String::from_utf8_lossy(&data[block_pos + 1..block_pos + 4])
                .trim()
                .to_string();
            match (block_type, name.as_str()) {
                (b'D', _) => match Self::parse_moment_block_bounded(data, block_pos, envelope_end) {
                    Ok(moment) => {
                        // Skip unknown/unrecognized moment types
                        if moment.product != RadarProduct::Unknown {
                            moments.push(moment);
                        }
                    }
                    Err(error) => {
                        if strict {
                            return Err(error);
                        }
                    }
                },
                (b'R', "VOL") => {
                    saw_vol = true;
                    match Self::parse_vol_block(data, block_pos, envelope_end) {
                        Ok(site) => vol_site = Some(site),
                        Err(error) => {
                            if strict {
                                return Err(error);
                            }
                        }
                    }
                }
                (b'R', "ELV") => {
                    saw_elv = true;
                    if let Err(error) = Self::parse_elv_block(data, block_pos, envelope_end) {
                        if strict {
                            return Err(error);
                        }
                    }
                }
                (b'R', "RAD") => {
                    saw_rad = true;
                    match Self::parse_rad_block(data, block_pos, envelope_end) {
                        Ok(value) => nyquist_velocity = value,
                        Err(error) => {
                            if strict {
                                return Err(error);
                            }
                        }
                    }
                }
                _ => {
                    if strict {
                        return Err(format!(
                            "corrupt Level-II volume: the Message-31 radial at offset \
                             {start_pos} carries a data block of type {:?} named {name:?}, \
                             which is neither a moment ('D') nor one of the three mandatory \
                             constant blocks VOL/ELV/RAD",
                            block_type as char
                        ));
                    }
                }
            }
        }

        if strict && !(saw_vol && saw_elv && saw_rad) {
            let mut absent = Vec::new();
            if !saw_vol {
                absent.push("VOL");
            }
            if !saw_elv {
                absent.push("ELV");
            }
            if !saw_rad {
                absent.push("RAD");
            }
            return Err(format!(
                "corrupt Level-II volume: the Message-31 radial at offset {start_pos} is \
                 missing the mandatory constant block(s) {absent:?}; without RAD there is no \
                 Nyquist velocity, and a velocity with no Nyquist has no alias test"
            ));
        }

        // Lenient keeps the historical floor of one legacy 2432-byte record
        // per step.  Strict steps by exactly what the message declared --
        // `radial_envelope_end` already proved that lands inside the volume
        // -- because rounding a short radial up to 2432 bytes walks into the
        // middle of the next one and drops whatever lay between.
        let msg_size_bytes = (msg_header.message_size as usize) * 2 + 12;
        let next_pos = if strict {
            start_pos + msg_size_bytes
        } else {
            start_pos + msg_size_bytes.max(2432)
        };
        if next_pos <= data.len() {
            cursor.set_position(next_pos as u64);
        }

        let radial = RadialData {
            azimuth: msg31.azimuth_angle,
            elevation: msg31.elevation_angle,
            azimuth_spacing: if msg31.azimuth_resolution == 1 {
                0.5
            } else {
                1.0
            },
            nyquist_velocity,
            radial_status: msg31.radial_status,
            moments,
        };

        Ok(MessageOutcome::Radial(
            msg31.elevation_number,
            msg31.cut_sector,
            vol_site,
            radial,
        ))
    }

    /// The last byte offset any of this radial's blocks may touch.
    ///
    /// The message header's size and the Message-31 header's radial length
    /// are two independent statements about the same extent.  Requiring
    /// them to agree would refuse volumes over a bookkeeping difference, so
    /// each is checked against the file on its own and the tighter of the
    /// survivors wins.  In strict mode a declaration that runs past the end
    /// of the volume, or is too small to hold the header it just supplied,
    /// is a refusal rather than a fallback to "the rest of the file".
    fn radial_envelope_end(
        data: &[u8],
        start_pos: usize,
        msg31_start: usize,
        msg_header: &MessageHeader,
        msg31: &Message31Header,
        radial_header_bytes: usize,
        strict: bool,
    ) -> Result<usize, String> {
        let mut envelope_end = data.len();

        let declared_msg_bytes = msg_header.message_size as usize * 2;
        if declared_msg_bytes > MSG_HEADER_SIZE {
            let end = start_pos + 12 + declared_msg_bytes;
            if end <= data.len() {
                envelope_end = envelope_end.min(end);
            } else if strict {
                return Err(format!(
                    "corrupt Level-II volume: the message at offset {start_pos} declares \
                     {declared_msg_bytes} bytes, which runs {} bytes past the end of the \
                     {}-byte volume",
                    end - data.len(),
                    data.len()
                ));
            }
        } else if strict {
            return Err(format!(
                "corrupt Level-II volume: the message at offset {start_pos} declares \
                 {declared_msg_bytes} bytes, too few to hold even its own {MSG_HEADER_SIZE}-byte \
                 header"
            ));
        }

        let radial_length = msg31.radial_length as usize;
        if radial_length >= radial_header_bytes {
            let end = msg31_start + radial_length;
            if end <= data.len() {
                envelope_end = envelope_end.min(end);
            } else if strict {
                return Err(format!(
                    "corrupt Level-II volume: the Message-31 radial at offset {start_pos} \
                     declares {radial_length} bytes, which runs {} bytes past the end of the \
                     {}-byte volume",
                    end - data.len(),
                    data.len()
                ));
            }
        } else if strict {
            return Err(format!(
                "corrupt Level-II volume: the Message-31 radial at offset {start_pos} declares \
                 a radial length of {radial_length} bytes, too few for its own \
                 {radial_header_bytes}-byte header and {} block pointers",
                msg31.data_block_count
            ));
        }

        if envelope_end < msg31_start + radial_header_bytes {
            return Err(format!(
                "corrupt Level-II volume: the Message-31 radial at offset {start_pos} has no \
                 room for its own {radial_header_bytes}-byte header inside the {} bytes that \
                 remain",
                envelope_end.saturating_sub(msg31_start)
            ));
        }
        Ok(envelope_end)
    }

    fn read_message_header(cursor: &mut Cursor<&Vec<u8>>) -> Result<MessageHeader, String> {
        let message_size = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let _rda_channel = cursor.read_u8().map_err(|e| e.to_string())?;
        let message_type = cursor.read_u8().map_err(|e| e.to_string())?;
        // Skip remaining 12 bytes of header
        let mut skip = [0u8; 12];
        cursor.read_exact(&mut skip).map_err(|e| e.to_string())?;
        Ok(MessageHeader {
            message_size,
            message_type,
        })
    }

    fn read_msg31_header(cursor: &mut Cursor<&Vec<u8>>) -> Result<Message31Header, String> {
        let mut radar_id = [0u8; 4];
        cursor
            .read_exact(&mut radar_id)
            .map_err(|e| e.to_string())?;
        let _collection_time = cursor.read_u32::<BigEndian>().map_err(|e| e.to_string())?;
        let _collection_date = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let _azimuth_number = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let azimuth_angle = cursor.read_f32::<BigEndian>().map_err(|e| e.to_string())?;
        let compression = cursor.read_u8().map_err(|e| e.to_string())?;
        let _spare = cursor.read_u8().map_err(|e| e.to_string())?;
        let radial_length = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let azimuth_resolution = cursor.read_u8().map_err(|e| e.to_string())?;
        let radial_status = cursor.read_u8().map_err(|e| e.to_string())?;
        let elevation_number = cursor.read_u8().map_err(|e| e.to_string())?;
        let cut_sector = cursor.read_u8().map_err(|e| e.to_string())?;
        let elevation_angle = cursor.read_f32::<BigEndian>().map_err(|e| e.to_string())?;
        let _spot_blanking = cursor.read_u8().map_err(|e| e.to_string())?;
        let _az_index_mode = cursor.read_u8().map_err(|e| e.to_string())?;
        let data_block_count = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;

        Ok(Message31Header {
            azimuth_angle,
            elevation_angle,
            elevation_number,
            azimuth_resolution,
            radial_status,
            cut_sector,
            data_block_count,
            compression,
            radial_length,
        })
    }

    /// Nyquist velocity from a Message-31 "RAD" (Radial) data block.
    ///
    /// The block is 28 bytes and says so in its own LRTUP field, which is
    /// how the layout below was settled against a real volume rather than
    /// against a reading of the ICD:
    ///
    /// ```text
    ///  0      block type       char       'R'
    ///  1..4   data name        char*3     "RAD"
    ///  4..6   LRTUP            int*2      28 -- the block's own length
    ///  6..8   unambiguous rng  int*2 /10  467.0 km
    ///  8..12  noise level H    real*4     -82.58 dBm
    /// 12..16  noise level V    real*4     -81.69 dBm
    /// 16..18  NYQUIST VELOCITY int*2 /100 8.27 m/s
    /// 18..20  radial flags     int*2
    /// 20..24  calib const H    real*4
    /// 24..28  calib const V    real*4
    /// ```
    ///
    /// Byte 26 -- the low half of the vertical calibration constant,
    /// reinterpreted as an unsigned integer -- was read here until
    /// 2026-07-30, which produced physically impossible Nyquist
    /// velocities: one KTLX volume reported 620.7, 450.8 and 313.9 m/s
    /// for cuts whose real values are 8-32 m/s.  Nothing in this
    /// workspace consumed the field, so the error was invisible until an
    /// observation pipeline needed it to decide which velocities might be
    /// aliased.  See VENDOR.md divergences 9 and 10.
    ///
    /// The version gate is LRTUP, not an allow-list of build numbers.  The
    /// block states how long it is, and both layouts this decoder has been
    /// checked against — the 20-byte Build 11.5/J block and the 28-byte
    /// Build 17/P and 24/AA block — put Nyquist at bytes 16..18.  A RAD
    /// block declaring any other length is a layout nothing here has read,
    /// so bytes 16..18 of it are an unknown quantity rather than a Nyquist
    /// velocity, and guessing is how a fabricated threshold licenses a
    /// folded velocity.
    fn parse_rad_block(data: &[u8], offset: usize, envelope_end: usize) -> Result<Option<f32>, String> {
        if offset + 6 > envelope_end {
            return Err(format!(
                "corrupt Level-II volume: the RAD block at offset {offset} has no room for its \
                 own length field inside the radial"
            ));
        }
        let lrtup = u16::from_be_bytes([data[offset + 4], data[offset + 5]]) as usize;
        if lrtup != 20 && lrtup != 28 {
            return Err(format!(
                "corrupt Level-II volume: the RAD block at offset {offset} declares LRTUP \
                 {lrtup}; only the 20-byte and 28-byte layouts have been verified to carry the \
                 Nyquist velocity at bytes 16..18, and reading an unverified layout there \
                 produces a number, not a measurement"
            ));
        }
        if offset + lrtup > envelope_end {
            return Err(format!(
                "corrupt Level-II volume: the RAD block at offset {offset} declares {lrtup} \
                 bytes but only {} remain inside the radial",
                envelope_end.saturating_sub(offset)
            ));
        }
        let nyquist_raw = u16::from_be_bytes([data[offset + 16], data[offset + 17]]);
        if nyquist_raw == 0 {
            return Ok(None);
        }
        Ok(Some(nyquist_raw as f32 / 100.0))
    }

    /// Validate a Message-31 "VOL" (Volume) constant block.
    ///
    /// ```text
    ///  0      block type       char       'R'
    ///  1..4   data name        char*3     "VOL"
    ///  4..6   LRTUP            int*2      44 (version 2.0) or 52 (3.0)
    ///  6      version major    int*1
    ///  7      version minor    int*1
    ///  8..12  latitude         real*4     degrees north
    /// 12..16  longitude        real*4     degrees east
    /// 16..18  site height      int*2      m MSL
    /// 18..20  feedhorn height  int*2      m AGL
    /// 20..24  calibration const real*4
    /// 24..28  horiz SHV tx pwr real*4
    /// 28..32  vert  SHV tx pwr real*4
    /// 32..36  system ZDR       real*4
    /// 36..40  initial system DP real*4
    /// 40..42  VCP number       int*2
    /// 42..44  processing status int*2
    /// ```
    ///
    /// Until 2026-07-30 the block was recognised by name and nothing else:
    /// `saw_vol = true` and no byte of it was looked at.  A radial could
    /// therefore carry a VOL block of any length, declaring any version,
    /// including one whose fields sit somewhere other than where the table
    /// above puts them, and the strict parser called the volume conforming.
    ///
    /// So the layout is now checked, and the check has teeth: the
    /// `(major, minor, LRTUP)` triple must be one this decoder has read in a
    /// real volume ([`KNOWN_VOL_LAYOUTS`]), and the latitude and longitude
    /// at 8..16 must then be a place on Earth.  Those two bounds are
    /// arithmetic rather than judgement -- a real radar has always satisfied
    /// them, and four bytes read at the wrong offset almost never do -- so
    /// they are what turns the version table from a label into a statement
    /// about where the fields are.
    fn parse_vol_block(data: &[u8], offset: usize, envelope_end: usize) -> Result<VolSite, String> {
        if offset + 8 > envelope_end {
            return Err(format!(
                "corrupt Level-II volume: the VOL block at offset {offset} has no room for its \
                 own length and version fields inside the radial"
            ));
        }
        let lrtup = u16::from_be_bytes([data[offset + 4], data[offset + 5]]) as usize;
        let major = data[offset + 6];
        let minor = data[offset + 7];
        let known_version = KNOWN_VOL_LAYOUTS
            .iter()
            .any(|(known_major, known_minor, _)| *known_major == major && *known_minor == minor);
        if !known_version {
            return Err(format!(
                "corrupt Level-II volume: the VOL block at offset {offset} declares generic \
                 format version {major}.{minor}; this decoder has only read {:?}, and the \
                 volume header fields of an unread version are at offsets nothing here has \
                 checked",
                KNOWN_VOL_LAYOUTS
                    .iter()
                    .map(|(major, minor, _)| format!("{major}.{minor}"))
                    .collect::<Vec<_>>()
            ));
        }
        let expected = KNOWN_VOL_LAYOUTS
            .iter()
            .find(|(known_major, known_minor, _)| *known_major == major && *known_minor == minor)
            .map(|(_, _, lrtup)| *lrtup)
            .expect("the version was just matched against this table");
        if lrtup != expected {
            return Err(format!(
                "corrupt Level-II volume: the VOL block at offset {offset} declares generic \
                 format version {major}.{minor} in {lrtup} bytes; every real volume of that \
                 version is {expected} bytes. The version and the length are two statements \
                 about where the same fields sit, and they disagree"
            ));
        }
        if offset + lrtup > envelope_end {
            return Err(format!(
                "corrupt Level-II volume: the VOL block at offset {offset} declares {lrtup} \
                 bytes but only {} remain inside the radial",
                envelope_end.saturating_sub(offset)
            ));
        }
        debug_assert!(lrtup >= VOL_READ_BYTES);
        let latitude = f32::from_be_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]);
        let longitude = f32::from_be_bytes([
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
            data[offset + 15],
        ]);
        if !latitude.is_finite() || !(-90.0..=90.0).contains(&latitude) {
            return Err(format!(
                "corrupt Level-II volume: the VOL block at offset {offset} puts the radar at \
                 latitude {latitude}, which is not a latitude. Either the block is not the \
                 {lrtup}-byte version {major}.{minor} layout it claims, or its contents are \
                 not the volume's"
            ));
        }
        if !longitude.is_finite() || !(-180.0..=180.0).contains(&longitude) {
            return Err(format!(
                "corrupt Level-II volume: the VOL block at offset {offset} puts the radar at \
                 longitude {longitude}, which is not a longitude. Either the block is not the \
                 {lrtup}-byte version {major}.{minor} layout it claims, or its contents are \
                 not the volume's"
            ));
        }
        // ICD 2620002 table XVII: site height (int*2, metres MSL) at 16,
        // feedhorn height (unsigned int*2, metres AGL) at 18.  The KTLX
        // survey fixture reads 370 m / 19 m here.  Bounds are arithmetic,
        // not judgement: no WSR-88D site sits below the Dead Sea or above
        // 4500 m, and no feedhorn tower is 200 m tall — four bytes read at
        // the wrong offset almost never satisfy both.
        let site_height = i16::from_be_bytes([data[offset + 16], data[offset + 17]]);
        let feedhorn_height = u16::from_be_bytes([data[offset + 18], data[offset + 19]]);
        if !(-450..=4500).contains(&site_height) {
            return Err(format!(
                "corrupt Level-II volume: the VOL block at offset {offset} puts the site \
                 ground at {site_height} m MSL, which is not a radar site on this planet. \
                 Either the block is not the {lrtup}-byte version {major}.{minor} layout it \
                 claims, or its contents are not the volume's"
            ));
        }
        if feedhorn_height > 200 {
            return Err(format!(
                "corrupt Level-II volume: the VOL block at offset {offset} puts the feedhorn \
                 {feedhorn_height} m above the site; real WSR-88D towers run roughly 3-50 m. \
                 Either the block is not the {lrtup}-byte version {major}.{minor} layout it \
                 claims, or its contents are not the volume's"
            ));
        }
        Ok(VolSite {
            latitude_deg: latitude,
            longitude_deg: longitude,
            site_height_m: site_height,
            feedhorn_height_m: feedhorn_height,
        })
    }

    /// Validate a Message-31 "ELV" (Elevation) constant block.
    ///
    /// ```text
    ///  0      block type       char       'R'
    ///  1..4   data name        char*3     "ELV"
    ///  4..6   LRTUP            int*2      12 -- the block's own length
    ///  6..8   atmos attenuation int*2     dB/km * 1000, negative
    ///  8..12  calibration const real*4    dBZ
    /// ```
    ///
    /// Twelve bytes is the whole block and every surveyed volume says so.
    /// Like VOL, this was previously recognised by name alone.
    fn parse_elv_block(data: &[u8], offset: usize, envelope_end: usize) -> Result<(), String> {
        if offset + 6 > envelope_end {
            return Err(format!(
                "corrupt Level-II volume: the ELV block at offset {offset} has no room for its \
                 own length field inside the radial"
            ));
        }
        let lrtup = u16::from_be_bytes([data[offset + 4], data[offset + 5]]) as usize;
        if lrtup != ELV_LRTUP {
            return Err(format!(
                "corrupt Level-II volume: the ELV block at offset {offset} declares LRTUP \
                 {lrtup}; the block is {ELV_LRTUP} bytes -- type, name, length, atmospheric \
                 attenuation and calibration constant -- and no other arrangement of those \
                 fields fits"
            ));
        }
        if offset + lrtup > envelope_end {
            return Err(format!(
                "corrupt Level-II volume: the ELV block at offset {offset} declares {lrtup} \
                 bytes but only {} remain inside the radial",
                envelope_end.saturating_sub(offset)
            ));
        }
        Ok(())
    }

    fn parse_moment_block_bounded(
        data: &[u8],
        offset: usize,
        envelope_end: usize,
    ) -> Result<MomentData, String> {
        if offset + MOMENT_HEADER_SIZE > envelope_end {
            return Err(format!(
                "corrupt Level-II volume: the moment block at offset {offset} has no room for \
                 its {MOMENT_HEADER_SIZE}-byte header inside the radial ({} bytes remain)",
                envelope_end.saturating_sub(offset)
            ));
        }
        // Header first, so the gate span can be checked before it is read.
        let gate_count = u16::from_be_bytes([data[offset + 8], data[offset + 9]]) as usize;
        let word_size = u16::from_be_bytes([data[offset + 18], data[offset + 19]]);
        if word_size != 8 && word_size != 16 {
            return Err(format!(
                "corrupt Level-II volume: the moment block at offset {offset} declares a \
                 {word_size}-bit data word; Level-II moments are 8 or 16 bits"
            ));
        }
        let span = gate_count * (word_size as usize / 8);
        if offset + MOMENT_HEADER_SIZE + span > envelope_end {
            return Err(format!(
                "corrupt Level-II volume: the moment block at offset {offset} declares \
                 {gate_count} gates of {word_size} bits ({span} bytes) but only {} remain \
                 inside the radial; those gates would be read out of a neighbouring radial",
                envelope_end
                    .saturating_sub(offset)
                    .saturating_sub(MOMENT_HEADER_SIZE)
            ));
        }
        Self::parse_moment_block(&data[..envelope_end], offset)
    }

    fn parse_moment_block(data: &[u8], offset: usize) -> Result<MomentData, String> {
        if offset + 28 > data.len() {
            return Err("Moment block too short".into());
        }

        let mut cursor = Cursor::new(&data[offset..]);
        let _block_type = cursor.read_u8().map_err(|e| e.to_string())?;
        let mut name_bytes = [0u8; 3];
        cursor
            .read_exact(&mut name_bytes)
            .map_err(|e| e.to_string())?;
        let name = String::from_utf8_lossy(&name_bytes).trim().to_string();

        let _reserved = cursor.read_u32::<BigEndian>().map_err(|e| e.to_string())?;
        let gate_count = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let first_gate_range = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let gate_size = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let _tover = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let _snr = cursor.read_u8().map_err(|e| e.to_string())?;
        let _flags = cursor.read_u8().map_err(|e| e.to_string())?;
        let data_word_size = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let scale = cursor.read_f32::<BigEndian>().map_err(|e| e.to_string())?;
        let offset_val = cursor.read_f32::<BigEndian>().map_err(|e| e.to_string())?;

        let product = RadarProduct::from_name(&name);

        let mut decoded = Vec::with_capacity(gate_count as usize);
        let mut censored = Vec::with_capacity(gate_count as usize);
        for _ in 0..gate_count {
            let raw = if data_word_size >= 16 {
                cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())? as u32
            } else {
                cursor.read_u8().map_err(|e| e.to_string())? as u32
            };
            // `value` is bit-for-bit what this decoder has always produced:
            // raw 0 and raw 1 are both NAN, everything else is the same
            // affine decode.  The arm that used to be `if raw <= 1` is split
            // only so the *reason* can be recorded beside it; see `censor`.
            let (value, code) = match raw {
                0 => (f32::NAN, censor::BELOW_THRESHOLD),
                1 => (f32::NAN, censor::RANGE_FOLDED),
                _ => ((raw as f32 - offset_val) / scale, censor::MEASURED),
            };
            decoded.push(value);
            censored.push(code);
        }

        Ok(MomentData {
            product,
            gate_count,
            first_gate_range,
            gate_size,
            data: decoded,
            censor: censored,
        })
    }

    /// Split raw radials into sweeps using radial_status and elevation_number.
    /// Exposed for testing.
    #[doc(hidden)]
    pub fn split_radials_into_sweeps(radials: Vec<(u8, u8, RadialData)>) -> Vec<Level2Sweep> {
        let mut sweeps: Vec<Level2Sweep> = Vec::new();
        let mut current_radials: Vec<RadialData> = Vec::new();
        let mut current_elev_num: u8 = 0;
        let mut current_cut_sector: u8 = 0;
        let mut sweep_counter: u16 = 0;

        fn flush(
            sweeps: &mut Vec<Level2Sweep>,
            radials: &mut Vec<RadialData>,
            elev_num: u8,
            cut_sector: u8,
            sweep_index: &mut u16,
        ) {
            if radials.is_empty() {
                return;
            }
            let elev_angle = radials[0].elevation;
            let reported: Vec<f32> = radials.iter().filter_map(|r| r.nyquist_velocity).collect();
            let nyquist = reported
                .iter()
                .copied()
                .fold(None::<f32>, |acc, value| Some(acc.map_or(value, |a| a.min(value))));
            let disagree = reported
                .iter()
                .any(|value| Some(*value) != nyquist)
                || reported.len() != radials.len();
            let start_status = radials[0].radial_status;
            let end_status = radials.last().map(|r| r.radial_status).unwrap_or(0xFF);
            sweeps.push(Level2Sweep {
                elevation_number: elev_num,
                elevation_angle: elev_angle,
                nyquist_velocity: nyquist,
                nyquist_radials_disagree: disagree,
                sweep_index: *sweep_index,
                start_status,
                end_status,
                cut_sector,
                radials: std::mem::take(radials),
            });
            *sweep_index += 1;
        }

        for (elev_num, cut_sector, radial) in radials {
            let is_status_start = matches!(radial.radial_status, 0 | 3 | 5);
            let is_elev_change = !current_radials.is_empty() && elev_num != current_elev_num;
            let should_split = is_status_start || is_elev_change;

            if should_split && !current_radials.is_empty() {
                flush(
                    &mut sweeps,
                    &mut current_radials,
                    current_elev_num,
                    current_cut_sector,
                    &mut sweep_counter,
                );
            }

            current_elev_num = elev_num;
            current_cut_sector = cut_sector;
            current_radials.push(radial);
        }

        flush(
            &mut sweeps,
            &mut current_radials,
            current_elev_num,
            current_cut_sector,
            &mut sweep_counter,
        );

        sweeps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- a synthetic Message-31 volume, assembled byte by byte -------------
    //
    // Every strict refusal below is one field of a *conforming* radial
    // changed, so the test says which single declaration made the volume
    // unsafe rather than asserting that a pile of junk fails.

    fn generic_block(name: &[u8; 4], lrtup: u16, body: &[u8]) -> Vec<u8> {
        let mut block = Vec::from(&name[..]);
        block.extend_from_slice(&lrtup.to_be_bytes());
        block.extend_from_slice(body);
        block.resize(lrtup as usize, 0);
        block
    }

    /// Verbatim VOL block of KTLX 2019-05-20 00:00:34Z, generic format
    /// version 2.0 in 44 bytes.  Decodes to 35.33336 N, 97.27776 W, site
    /// 370 m, feedhorn 19 m, VCP 32 -- which is KTLX.
    const REAL_VOL_V2: [u8; 44] = [
        0x52, 0x56, 0x4f, 0x4c, // 'R' "VOL"
        0x00, 0x2c, // LRTUP = 44
        0x02, 0x00, // version 2.0
        0x42, 0x0d, 0x55, 0x5d, // latitude   35.33336
        0xc2, 0xc2, 0x8e, 0x37, // longitude -97.27776
        0x01, 0x72, // site height   370 m
        0x00, 0x13, // feedhorn       19 m
        0xc2, 0x34, 0x4a, 0x62, // calibration constant
        0x43, 0x81, 0x2e, 0x7c, // horizontal SHV tx power
        0x43, 0x6f, 0x43, 0x73, // vertical   SHV tx power
        0x3d, 0xe5, 0x4f, 0x09, // system differential reflectivity
        0x42, 0x70, 0x00, 0x00, // initial system differential phase
        0x00, 0x20, // VCP 32
        0x00, 0x01, // processing status
    ];

    /// Verbatim VOL block of KTLX 2013-05-20 19:51:11Z -- the Moore
    /// tornado volume, and the era the pre-2016 `.gz` keys cover.  Generic
    /// format version 1.0 in 44 bytes: 35.33306 N, 97.27748 W, site 369 m,
    /// VCP 12.  The site coordinates differ in the fifth decimal from the
    /// 2015-onward volumes because the RDA's own record of them was
    /// refined, which is a nice independent check that these bytes are read
    /// at the right offsets rather than copied from a later file.
    const REAL_VOL_V1: [u8; 44] = [
        0x52, 0x56, 0x4f, 0x4c, // 'R' "VOL"
        0x00, 0x2c, // LRTUP = 44
        0x01, 0x00, // version 1.0
        0x42, 0x0d, 0x55, 0x0d, // latitude   35.33306
        0xc2, 0xc2, 0x8e, 0x12, // longitude -97.27748
        0x01, 0x71, // site height   369 m
        0x00, 0x13, // feedhorn       19 m
        0xc2, 0x2f, 0xa8, 0xf4, // calibration constant
        0x43, 0x39, 0x0e, 0x8e, // horizontal SHV tx power
        0x43, 0x35, 0x74, 0xa2, // vertical   SHV tx power
        0xbd, 0xa8, 0x0d, 0xcb, // system differential reflectivity
        0x41, 0xc8, 0x00, 0x00, // initial system differential phase
        0x00, 0x0c, // VCP 12
        0x00, 0x00, // processing status
    ];

    /// Verbatim RAD block of that same 2013 radial: the 20-byte layout,
    /// Nyquist 8.30 m/s at bytes 16..18.  It sits beside the 1.0 VOL block
    /// in every volume of that era.
    const REAL_RAD_V1_ERA: [u8; 20] = [
        0x52, 0x52, 0x41, 0x44, // 'R' "RAD"
        0x00, 0x14, // LRTUP = 20
        0x12, 0x34, // unambiguous range 466.0 km
        0xc2, 0x9f, 0xcf, 0xd7, // noise level H
        0xc2, 0x9f, 0x57, 0xb6, // noise level V
        0x03, 0x3e, // Nyquist 830 -> 8.30 m/s
        0x00, 0x00, // radial flags
    ];

    /// Verbatim VOL block of KFWS 2026-07-29 00:04:22Z, version 3.0 in 52
    /// bytes: the same 44-byte layout with eight bytes appended.  Decodes to
    /// 32.57293 N, 97.30313 W, site 212 m, VCP 35 -- which is KFWS.
    const REAL_VOL_V3: [u8; 52] = [
        0x52, 0x56, 0x4f, 0x4c, // 'R' "VOL"
        0x00, 0x34, // LRTUP = 52
        0x03, 0x00, // version 3.0
        0x42, 0x02, 0x4a, 0xc1, // latitude   32.57293
        0xc2, 0xc2, 0x9b, 0x36, // longitude -97.30313
        0x00, 0xd4, // site height   212 m
        0x00, 0x18, // feedhorn       24 m
        0xc2, 0x28, 0x40, 0xf2, // calibration constant
        0x43, 0x5f, 0xf1, 0x1d, // horizontal SHV tx power
        0x43, 0x6b, 0xb3, 0xa3, // vertical   SHV tx power
        0xbf, 0xaa, 0x9c, 0x1a, // system differential reflectivity
        0x42, 0x70, 0x00, 0x00, // initial system differential phase
        0x00, 0x23, // VCP 35
        0x00, 0x03, // processing status
        0x00, 0x00, 0x00, 0x00, // version 3.0 tail
        0x00, 0x00, 0x00, 0x00,
    ];

    /// Verbatim ELV block of the same KTLX 2019 radial: atmospheric
    /// attenuation -0.012 dB/km, calibration constant -43.8125 dBZ.
    const REAL_ELV: [u8; 12] = [
        0x52, 0x45, 0x4c, 0x56, // 'R' "ELV"
        0x00, 0x0c, // LRTUP = 12
        0xff, 0xf4, // atmospheric attenuation -12
        0xc2, 0x2f, 0x40, 0x00, // calibration constant
    ];

    fn vol_block() -> Vec<u8> {
        REAL_VOL_V2.to_vec()
    }

    fn elv_block() -> Vec<u8> {
        REAL_ELV.to_vec()
    }

    fn rad_block(nyquist_hundredths: u16) -> Vec<u8> {
        let mut body = vec![0u8; 10]; // unambiguous range + two noise levels
        body.extend_from_slice(&nyquist_hundredths.to_be_bytes());
        generic_block(b"RRAD", 28, &body)
    }

    fn ref_moment(gates: u16) -> Vec<u8> {
        let mut block = Vec::from(&b"DREF"[..]);
        block.extend_from_slice(&0u32.to_be_bytes()); // reserved
        block.extend_from_slice(&gates.to_be_bytes());
        block.extend_from_slice(&2125u16.to_be_bytes()); // first gate range
        block.extend_from_slice(&250u16.to_be_bytes()); // gate size
        block.extend_from_slice(&0u16.to_be_bytes()); // tover
        block.push(0); // snr threshold
        block.push(0); // flags
        block.extend_from_slice(&8u16.to_be_bytes()); // data word size
        block.extend_from_slice(&2.0f32.to_be_bytes()); // scale
        block.extend_from_slice(&66.0f32.to_be_bytes()); // offset
        assert_eq!(block.len(), MOMENT_HEADER_SIZE);
        // raw 2..  ->  (raw - 66) / 2 dBZ
        block.extend((0..gates).map(|g| (100 + g) as u8));
        block
    }

    /// `ref_moment` with the gate words written out by hand, so a test can
    /// place raw 0 and raw 1 exactly where it wants them.
    fn ref_moment_raw(words: &[u8]) -> Vec<u8> {
        let mut block = ref_moment(words.len() as u16);
        block.truncate(MOMENT_HEADER_SIZE);
        block.extend_from_slice(words);
        block
    }

    #[derive(Clone)]
    struct RadialBuilder {
        compression: u8,
        radial_length: Option<u16>,
        message_size: Option<u16>,
        pointers: Option<Vec<u32>>,
        blocks: Vec<Vec<u8>>,
        radial_status: u8,
    }

    impl RadialBuilder {
        fn conforming() -> Self {
            Self {
                compression: 0,
                radial_length: None,
                message_size: None,
                pointers: None,
                blocks: vec![vol_block(), elv_block(), rad_block(2384), ref_moment(4)],
                radial_status: 3,
            }
        }

        fn build(&self) -> Vec<u8> {
            let count = self.blocks.len();
            let header_bytes = MSG31_HEADER_SIZE + 4 * count;
            let mut natural = Vec::new();
            let mut running = header_bytes as u32;
            for block in &self.blocks {
                natural.push(running);
                running += block.len() as u32;
            }
            let pointers = self.pointers.clone().unwrap_or(natural);

            let mut msg31 = Vec::from(&b"KTLX"[..]);
            msg31.extend_from_slice(&0u32.to_be_bytes()); // collection time
            msg31.extend_from_slice(&20663u16.to_be_bytes()); // julian date
            msg31.extend_from_slice(&1u16.to_be_bytes()); // azimuth number
            msg31.extend_from_slice(&90.0f32.to_be_bytes()); // azimuth angle
            msg31.push(self.compression);
            msg31.push(0); // spare
            msg31.extend_from_slice(
                &self.radial_length.unwrap_or(running as u16).to_be_bytes(),
            );
            msg31.push(1); // azimuth resolution: 0.5 deg
            msg31.push(self.radial_status);
            msg31.push(1); // elevation number
            msg31.push(0); // cut sector
            msg31.extend_from_slice(&0.5f32.to_be_bytes()); // elevation angle
            msg31.push(0); // spot blanking
            msg31.push(0); // azimuth indexing mode
            msg31.extend_from_slice(&(count as u16).to_be_bytes());
            for pointer in &pointers {
                msg31.extend_from_slice(&pointer.to_be_bytes());
            }
            assert_eq!(msg31.len(), header_bytes);
            for block in &self.blocks {
                msg31.extend_from_slice(block);
            }

            let declared = self
                .message_size
                .unwrap_or(((MSG_HEADER_SIZE + msg31.len()) / 2) as u16);
            let mut message = vec![0u8; 12]; // CTM
            message.extend_from_slice(&declared.to_be_bytes());
            message.push(0); // rda channel
            message.push(31); // message type
            message.extend_from_slice(&[0u8; 12]);
            message.extend_from_slice(&msg31);
            message
        }
    }

    /// bzip2 exactly as the RDA does, so an LDM block in a fixture is an
    /// LDM block.  This used to store the messages uncompressed behind a
    /// length word, which no real volume does; the framing discriminator
    /// then correctly classified the fixture as *unframed* and the tests
    /// were suddenly exercising a shape they did not mean to.
    fn bzip2_block(payload: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut encoder =
            bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::fast());
        encoder.write_all(payload).unwrap();
        let compressed = encoder.finish().unwrap();
        assert_eq!(&compressed[..3], b"BZh", "an LDM block is a bzip2 stream");
        compressed
    }

    fn archive2_volume(messages: &[Vec<u8>]) -> Vec<u8> {
        let mut raw = Vec::from(&b"AR2V0006."[..]);
        raw.resize(VOLUME_HEADER_SIZE, 0);
        raw[14..16].copy_from_slice(&20663u16.to_be_bytes());
        raw[16..20].copy_from_slice(&72_196_232u32.to_be_bytes());
        raw[20..24].copy_from_slice(b"KTLX");
        let block = bzip2_block(&messages.concat());
        raw.extend_from_slice(&(block.len() as i32).to_be_bytes());
        raw.extend_from_slice(&block);
        raw
    }

    fn strict_error(builder: &RadialBuilder) -> String {
        let raw = archive2_volume(&[builder.build()]);
        Level2File::parse_strict(&raw).expect_err("strict must refuse this volume")
    }

    #[test]
    fn a_conforming_message31_volume_decodes_under_both_modes() {
        let raw = archive2_volume(&[RadialBuilder::conforming().build()]);
        for file in [
            Level2File::parse_strict(&raw).unwrap(),
            Level2File::parse(&raw).unwrap(),
        ] {
            assert_eq!(file.station_id, "KTLX");
            assert_eq!(file.sweeps.len(), 1);
            let radial = &file.sweeps[0].radials[0];
            assert_eq!(radial.nyquist_velocity, Some(23.84));
            assert_eq!(radial.moments.len(), 1);
            let moment = &radial.moments[0];
            assert_eq!(moment.product, RadarProduct::Reflectivity);
            assert_eq!(moment.gate_count, 4);
            assert_eq!(moment.first_gate_range, 2125);
            assert_eq!(moment.gate_size, 250);
            // (100 - 66) / 2 == 17 dBZ, and it climbs by 0.5 per gate.
            assert_eq!(moment.data, vec![17.0, 17.5, 18.0, 18.5]);
        }
    }

    #[test]
    fn a_block_pointer_that_leaves_its_own_radial_is_refused() {
        // The pointer is inside the decompressed volume -- the old bound --
        // but past the end of this radial, so following it reads the next
        // radial's bytes as this one's.
        let mut builder = RadialBuilder::conforming();
        let mut pointers: Vec<u32> = {
            let count = builder.blocks.len();
            let mut running = (MSG31_HEADER_SIZE + 4 * count) as u32;
            builder
                .blocks
                .iter()
                .map(|block| {
                    let at = running;
                    running += block.len() as u32;
                    at
                })
                .collect()
        };
        *pointers.last_mut().unwrap() += 4096;
        builder.pointers = Some(pointers);
        let err = strict_error(&builder);
        assert!(err.contains("outside its own"), "{err}");
        assert!(err.contains("neighbouring radial"), "{err}");

        // A pointer aimed back into the radial's own header is equally out
        // of contract, and is not a "moment absent" zero.
        let mut builder = RadialBuilder::conforming();
        builder.pointers = Some(vec![48, 48, 48, 8]);
        assert!(strict_error(&builder).contains("outside its own"));
    }

    #[test]
    fn a_generic_block_that_is_not_rad_is_no_longer_a_nyquist_source() {
        // Every block whose first byte was 'R' used to overwrite the
        // radial's Nyquist velocity, so a VOL or ELV block reached by a
        // stray pointer supplied one.  Order the mandatory blocks so that
        // ELV, not RAD, is the last 'R' block seen.
        let mut builder = RadialBuilder::conforming();
        builder.blocks = vec![vol_block(), rad_block(2384), elv_block(), ref_moment(4)];
        let raw = archive2_volume(&[builder.build()]);
        let file = Level2File::parse_strict(&raw).unwrap();
        assert_eq!(file.sweeps[0].radials[0].nyquist_velocity, Some(23.84));

        // And a block named neither VOL, ELV, RAD nor a moment is refused
        // rather than silently consulted.
        let mut builder = RadialBuilder::conforming();
        builder.blocks = vec![
            vol_block(),
            elv_block(),
            rad_block(2384),
            generic_block(b"RXYZ", 20, &[]),
        ];
        let err = strict_error(&builder);
        assert!(err.contains("\"XYZ\""), "{err}");
        assert!(err.contains("VOL/ELV/RAD"), "{err}");
    }

    #[test]
    fn a_radial_missing_a_mandatory_constant_block_is_refused() {
        let mut builder = RadialBuilder::conforming();
        builder.blocks = vec![vol_block(), elv_block(), ref_moment(4)];
        let err = strict_error(&builder);
        assert!(err.contains("mandatory constant block"), "{err}");
        assert!(err.contains("RAD"), "{err}");
        assert!(err.contains("no alias test"), "{err}");
    }

    #[test]
    fn a_declared_length_that_leaves_the_volume_is_refused() {
        let mut builder = RadialBuilder::conforming();
        builder.radial_length = Some(60_000);
        let err = strict_error(&builder);
        assert!(err.contains("declares 60000 bytes"), "{err}");
        assert!(err.contains("past the end"), "{err}");

        let mut builder = RadialBuilder::conforming();
        builder.radial_length = Some(8);
        let err = strict_error(&builder);
        assert!(err.contains("too few for its own"), "{err}");

        let mut builder = RadialBuilder::conforming();
        builder.message_size = Some(30_000);
        let err = strict_error(&builder);
        assert!(err.contains("past the end"), "{err}");

        let mut builder = RadialBuilder::conforming();
        builder.message_size = Some(4);
        let err = strict_error(&builder);
        assert!(err.contains("too few to hold even its own"), "{err}");
    }

    #[test]
    fn a_moment_whose_gates_leave_the_radial_is_refused() {
        // The header says 20,000 gates; the radial has room for four.
        let mut builder = RadialBuilder::conforming();
        let mut moment = ref_moment(4);
        moment[8..10].copy_from_slice(&20_000u16.to_be_bytes());
        builder.blocks = vec![vol_block(), elv_block(), rad_block(2384), moment];
        let err = strict_error(&builder);
        assert!(err.contains("20000 gates"), "{err}");
        assert!(err.contains("neighbouring radial"), "{err}");

        // A data word width Level-II does not define is refused before any
        // gate is read, because the span cannot even be computed from it.
        let mut builder = RadialBuilder::conforming();
        let mut moment = ref_moment(4);
        moment[18..20].copy_from_slice(&32u16.to_be_bytes());
        builder.blocks = vec![vol_block(), elv_block(), rad_block(2384), moment];
        assert!(strict_error(&builder).contains("32-bit data word"));
    }

    #[test]
    fn a_compressed_radial_is_refused_rather_than_read_as_plain_bytes() {
        let mut builder = RadialBuilder::conforming();
        builder.compression = 1;
        let err = strict_error(&builder);
        assert!(err.contains("compression indicator 1"), "{err}");
        assert!(err.contains("wrong range"), "{err}");
    }

    #[test]
    fn an_undecodable_bzip2_block_never_becomes_radial_bytes() {
        // "BZh" framing with a garbage payload: the block used to be
        // substituted into the message stream still compressed.
        let mut raw = Vec::from(&b"AR2V0006."[..]);
        raw.resize(VOLUME_HEADER_SIZE, 0);
        raw[20..24].copy_from_slice(b"KTLX");
        let good = bzip2_block(&RadialBuilder::conforming().build());
        raw.extend_from_slice(&(good.len() as i32).to_be_bytes());
        raw.extend_from_slice(&good);
        let junk = b"BZh9-not-actually-a-bzip2-stream";
        raw.extend_from_slice(&(junk.len() as i32).to_be_bytes());
        raw.extend_from_slice(junk);

        let err = Level2File::parse_strict(&raw).unwrap_err();
        assert!(err.contains("will not decompress"), "{err}");

        // Lenient keeps the volume, but the undecodable block contributes
        // nothing rather than being parsed as though it were radial bytes.
        let file = Level2File::parse(&raw).unwrap();
        assert_eq!(file.sweeps.len(), 1);
        assert_eq!(file.sweeps[0].radials.len(), 1);
    }

    fn make_radial(azimuth: f32, elevation: f32, status: u8) -> RadialData {
        RadialData {
            azimuth,
            elevation,
            azimuth_spacing: 1.0,
            nyquist_velocity: None,
            radial_status: status,
            moments: Vec::new(),
        }
    }

    #[test]
    fn test_normal_cuts_split_correctly() {
        // Normal VCP: 3 tilts at 0.5°, 0.9°, 1.3°
        let radials = vec![
            // Tilt 1: elev_num=1, 0.5°
            (1, 0, make_radial(0.0, 0.5, 3)), // start volume
            (1, 0, make_radial(1.0, 0.5, 1)),
            (1, 0, make_radial(2.0, 0.5, 2)), // end elev
            // Tilt 2: elev_num=2, 0.9°
            (2, 0, make_radial(0.0, 0.9, 0)), // start elev
            (2, 0, make_radial(1.0, 0.9, 1)),
            (2, 0, make_radial(2.0, 0.9, 2)), // end elev
            // Tilt 3: elev_num=3, 1.3°
            (3, 0, make_radial(0.0, 1.3, 0)), // start elev
            (3, 0, make_radial(1.0, 1.3, 1)),
            (3, 0, make_radial(2.0, 1.3, 4)), // end volume
        ];

        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(sweeps.len(), 3);
        assert_eq!(sweeps[0].elevation_number, 1);
        assert_eq!(sweeps[1].elevation_number, 2);
        assert_eq!(sweeps[2].elevation_number, 3);
        assert_eq!(sweeps[0].radials.len(), 3);
        assert_eq!(sweeps[1].radials.len(), 3);
        assert_eq!(sweeps[2].radials.len(), 3);
        // Verify sweep_index
        assert_eq!(sweeps[0].sweep_index, 0);
        assert_eq!(sweeps[1].sweep_index, 1);
        assert_eq!(sweeps[2].sweep_index, 2);
        // Verify start/end status
        assert_eq!(sweeps[0].start_status, 3);
        assert_eq!(sweeps[0].end_status, 2);
        assert_eq!(sweeps[2].end_status, 4);
    }

    #[test]
    fn test_sails_duplicate_cuts_split() {
        // SAILS: tilt 1 (0.5°), tilt 2 (0.9°), SAILS repeat of tilt 1 (0.5°), tilt 3 (1.3°)
        let radials = vec![
            // First 0.5° pass: elev_num=1
            (1, 0, make_radial(0.0, 0.5, 3)),
            (1, 0, make_radial(1.0, 0.5, 1)),
            (1, 0, make_radial(2.0, 0.5, 2)),
            // 0.9°: elev_num=2
            (2, 0, make_radial(0.0, 0.9, 0)),
            (2, 0, make_radial(1.0, 0.9, 1)),
            (2, 0, make_radial(2.0, 0.9, 2)),
            // SAILS repeat 0.5°: elev_num=1 again
            (1, 0, make_radial(0.0, 0.5, 0)),
            (1, 0, make_radial(1.0, 0.5, 1)),
            (1, 0, make_radial(2.0, 0.5, 2)),
            // 1.3°: elev_num=3
            (3, 0, make_radial(0.0, 1.3, 0)),
            (3, 0, make_radial(1.0, 1.3, 1)),
            (3, 0, make_radial(2.0, 1.3, 4)),
        ];

        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(
            sweeps.len(),
            4,
            "SAILS repeat should produce 4 sweeps, not 3"
        );
        assert_eq!(sweeps[0].elevation_number, 1);
        assert_eq!(sweeps[1].elevation_number, 2);
        assert_eq!(sweeps[2].elevation_number, 1); // SAILS repeat
        assert_eq!(sweeps[3].elevation_number, 3);
        assert_eq!(sweeps[2].radials.len(), 3);
    }

    #[test]
    fn test_status_5_starts_new_sweep() {
        // Status 5 (start elev mid-volume) should also split
        let radials = vec![
            (1, 0, make_radial(0.0, 0.5, 3)),
            (1, 0, make_radial(1.0, 0.5, 1)),
            (1, 0, make_radial(2.0, 0.5, 2)),
            // Status 5 sweep start
            (2, 0, make_radial(0.0, 0.9, 5)),
            (2, 0, make_radial(1.0, 0.9, 1)),
            (2, 0, make_radial(2.0, 0.9, 2)),
        ];

        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(sweeps.len(), 2);
        assert_eq!(sweeps[1].start_status, 5);
    }

    #[test]
    fn test_elevation_change_fallback_without_status_marker() {
        // If status markers are missing (all status=1), elevation_number change splits
        let radials = vec![
            (1, 0, make_radial(0.0, 0.5, 1)),
            (1, 0, make_radial(1.0, 0.5, 1)),
            (2, 0, make_radial(0.0, 0.9, 1)), // elev change but no status marker
            (2, 0, make_radial(1.0, 0.9, 1)),
        ];

        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(
            sweeps.len(),
            2,
            "elevation_number change should split even without status marker"
        );
        assert_eq!(sweeps[0].elevation_number, 1);
        assert_eq!(sweeps[1].elevation_number, 2);
    }

    #[test]
    fn nyquist_comes_from_a_real_radial_block() {
        // Verbatim bytes of a RAD block from KTLX 2026-07-28 20:03:16Z
        // (volume sha256 877cf512...63504c).  Every field decodes to a
        // physically sane number at these offsets, and the block's own
        // LRTUP says 28, which is exactly where the layout ends -- that
        // is the proof the offsets are right, not a citation.
        let block: [u8; 28] = [
            0x52, 0x52, 0x41, 0x44, // block type 'R', name "RAD"
            0x00, 0x1c, // LRTUP = 28
            0x12, 0x3e, // unambiguous range 4670 -> 467.0 km
            0xc2, 0xa5, 0x26, 0x9d, // noise level H  -82.58 dBm
            0xc2, 0xa3, 0x62, 0x24, // noise level V  -81.69 dBm
            0x03, 0x3b, // Nyquist 827 -> 8.27 m/s
            0x00, 0x00, // radial flags
            0xc2, 0x29, 0xc8, 0x3a, // calibration constant H
            0xc2, 0x29, 0xb0, 0x18, // calibration constant V
        ];
        assert_eq!(
            u16::from_be_bytes([block[4], block[5]]) as usize,
            block.len(),
            "LRTUP must account for the whole block"
        );
        assert_eq!(
            Level2File::parse_rad_block(&block, 0, block.len()),
            Ok(Some(8.27))
        );
        // The retired byte 26 reads the calibration constant and yields
        // 450.8 m/s -- the signature of the bug this test pins shut.
        let retired = u16::from_be_bytes([block[26], block[27]]) as f32 / 100.0;
        assert!((retired - 450.8).abs() < 0.01, "retired offset gave {retired}");

        // Zero means absent, not zero.
        let mut absent = block;
        absent[16] = 0;
        absent[17] = 0;
        assert_eq!(Level2File::parse_rad_block(&absent, 0, absent.len()), Ok(None));
        // A block too short to hold the field is a refusal, never a guess
        // and never a panic.
        assert!(Level2File::parse_rad_block(&block[..17], 0, 17).is_err());
    }

    #[test]
    fn a_rad_block_of_an_unverified_length_is_refused_rather_than_read() {
        let mut block = [0u8; 28];
        block[..4].copy_from_slice(b"RRAD");
        // Nyquist bytes carry 23.84 m/s, a perfectly plausible number.
        block[16..18].copy_from_slice(&2384u16.to_be_bytes());

        // The 20-byte Build 11.5/J layout and the 28-byte Build 17/24
        // layout both put Nyquist at 16..18, so both are read.
        for lrtup in [20u16, 28u16] {
            block[4..6].copy_from_slice(&lrtup.to_be_bytes());
            assert_eq!(
                Level2File::parse_rad_block(&block, 0, block.len()),
                Ok(Some(23.84)),
                "LRTUP {lrtup}"
            );
        }

        // Anything else is a layout nothing here has read.  The bytes at
        // 16..18 still decode to 23.84 -- squarely inside the plausible
        // band -- which is exactly why believing them would be dangerous.
        for lrtup in [12u16, 24u16, 44u16, 0u16] {
            block[4..6].copy_from_slice(&lrtup.to_be_bytes());
            let err = Level2File::parse_rad_block(&block, 0, block.len()).unwrap_err();
            assert!(err.contains(&format!("declares LRTUP {lrtup}")), "{err}");
            assert!(err.contains("verified"), "{err}");
        }

        // A block whose declared length leaves the radial is refused even
        // when the length itself is one of the two known ones.
        block[4..6].copy_from_slice(&28u16.to_be_bytes());
        let err = Level2File::parse_rad_block(&block, 0, 20).unwrap_err();
        assert!(err.contains("declares 28 bytes but only 20 remain"), "{err}");
    }
    #[test]
    fn both_real_vol_layouts_decode_and_their_fields_are_where_the_table_says() {
        // Three real volumes spanning thirteen years: version 1.0 and 2.0
        // in 44 bytes, 3.0 in 52.  All must pass, and the fields the gate
        // leans on must be the radars they came from.
        for (block, lat, lon, vcp) in [
            (&REAL_VOL_V1[..], 35.33306_f32, -97.27748_f32, 12u16),
            (&REAL_VOL_V2[..], 35.33336_f32, -97.27776_f32, 32u16),
            (&REAL_VOL_V3[..], 32.57293_f32, -97.30313_f32, 35u16),
        ] {
            assert_eq!(
                u16::from_be_bytes([block[4], block[5]]) as usize,
                block.len(),
                "LRTUP must account for the whole block"
            );
            let site = Level2File::parse_vol_block(block, 0, block.len()).unwrap();
            assert!((site.latitude_deg - lat).abs() < 1e-4, "lat {}", site.latitude_deg);
            assert!((site.longitude_deg - lon).abs() < 1e-4, "lon {}", site.longitude_deg);
            // The two height fields are the reason the block is now
            // surfaced instead of merely validated: site ground at 16,
            // feedhorn AGL at 18, beam origin their sum.
            assert_eq!(
                site.site_height_m,
                i16::from_be_bytes([block[16], block[17]])
            );
            assert_eq!(
                site.feedhorn_height_m,
                u16::from_be_bytes([block[18], block[19]])
            );
            assert!((0..=200).contains(&site.feedhorn_height_m), "{site:?}");
            assert!(site.antenna_height_m() > f64::from(site.site_height_m));
            assert_eq!(u16::from_be_bytes([block[40], block[41]]), vcp);
        }
        assert_eq!(Level2File::parse_elv_block(&REAL_ELV, 0, REAL_ELV.len()), Ok(()));
    }

    #[test]
    fn a_vol_version_this_decoder_has_not_read_is_refused() {
        // Every field of the real 3.0 block stays put; only the version
        // changes.  The latitude and longitude still decode to KFWS, which
        // is exactly why a version check is needed: nothing downstream of
        // the version byte can tell that the layout is unknown.
        for (major, minor) in [(1u8, 1u8), (2, 1), (3, 1), (4, 0), (0, 0), (255, 255)] {
            let mut block = REAL_VOL_V3;
            block[6] = major;
            block[7] = minor;
            let err = Level2File::parse_vol_block(&block, 0, block.len()).unwrap_err();
            assert!(
                err.contains(&format!("version {major}.{minor}")),
                "{major}.{minor}: {err}"
            );
            assert!(err.contains("only read"), "{err}");
        }
    }

    #[test]
    fn a_vol_length_that_contradicts_its_version_is_refused() {
        // The two layouts swapped: 2.0 is a 44-byte block and 3.0 a 52-byte
        // one, so each declaring the other's length is stating two
        // incompatible things about where its fields are.
        let mut block = REAL_VOL_V2;
        block[4..6].copy_from_slice(&52u16.to_be_bytes());
        let err = Level2File::parse_vol_block(&block, 0, 52).unwrap_err();
        assert!(err.contains("version 2.0 in 52 bytes"), "{err}");
        assert!(err.contains("every real volume of that version is 44"), "{err}");

        let mut block = REAL_VOL_V1;
        block[4..6].copy_from_slice(&52u16.to_be_bytes());
        let err = Level2File::parse_vol_block(&block, 0, 52).unwrap_err();
        assert!(err.contains("version 1.0 in 52 bytes"), "{err}");

        let mut block = REAL_VOL_V3;
        block[4..6].copy_from_slice(&44u16.to_be_bytes());
        let err = Level2File::parse_vol_block(&block, 0, 52).unwrap_err();
        assert!(err.contains("version 3.0 in 44 bytes"), "{err}");

        // And lengths belonging to neither layout, including the RAD and
        // ELV block lengths, which are the plausible mix-ups.
        for lrtup in [12u16, 20, 28, 48, 0, 65535] {
            let mut block = REAL_VOL_V3;
            block[4..6].copy_from_slice(&lrtup.to_be_bytes());
            let err = Level2File::parse_vol_block(&block, 0, 52).unwrap_err();
            assert!(err.contains(&format!("in {lrtup} bytes")), "{lrtup}: {err}");
        }
    }

    #[test]
    fn a_vol_block_that_does_not_place_the_radar_on_earth_is_refused() {
        // The version and length agree; the bytes at 8..16 are not a
        // position.  That is what a correctly-labelled block of some other
        // layout looks like from here.
        for (label, lat_bytes) in [
            ("far north", [0x43, 0x16, 0x00, 0x00]),   // 150.0
            ("far south", [0xc3, 0x16, 0x00, 0x00]),   // -150.0
            ("not a number", [0x7f, 0xc0, 0x00, 0x00]), // NaN
            ("infinite", [0x7f, 0x80, 0x00, 0x00]),
        ] {
            let mut block = REAL_VOL_V3;
            block[8..12].copy_from_slice(&lat_bytes);
            let err = Level2File::parse_vol_block(&block, 0, 52).unwrap_err();
            assert!(err.contains("not a latitude"), "{label}: {err}");
        }
        for (label, lon_bytes) in [
            ("east of everywhere", [0x43, 0xfa, 0x00, 0x00]), // 500.0
            ("west of everywhere", [0xc3, 0xfa, 0x00, 0x00]),
            ("not a number", [0x7f, 0xc0, 0x00, 0x00]),
        ] {
            let mut block = REAL_VOL_V3;
            block[12..16].copy_from_slice(&lon_bytes);
            let err = Level2File::parse_vol_block(&block, 0, 52).unwrap_err();
            assert!(err.contains("not a longitude"), "{label}: {err}");
        }
    }

    #[test]
    fn an_elv_block_of_any_other_length_is_refused() {
        for lrtup in [0u16, 8, 11, 13, 20, 28, 44, 52, 65535] {
            let mut block = REAL_ELV;
            block[4..6].copy_from_slice(&lrtup.to_be_bytes());
            let err = Level2File::parse_elv_block(&block, 0, block.len()).unwrap_err();
            assert!(err.contains(&format!("declares LRTUP {lrtup}")), "{lrtup}: {err}");
        }
    }

    #[test]
    fn a_constant_block_that_leaves_the_radial_is_refused() {
        // Declared length inside the known set, but past the radial's end.
        let err = Level2File::parse_vol_block(&REAL_VOL_V3, 0, 40).unwrap_err();
        assert!(err.contains("declares 52 bytes but only 40 remain"), "{err}");
        let err = Level2File::parse_elv_block(&REAL_ELV, 0, 9).unwrap_err();
        assert!(err.contains("declares 12 bytes but only 9 remain"), "{err}");
        // Too short even to hold the length/version fields.
        assert!(Level2File::parse_vol_block(&REAL_VOL_V3, 0, 7).is_err());
        assert!(Level2File::parse_elv_block(&REAL_ELV, 0, 5).is_err());
    }

    /// The pre-2016 shape, assembled from the same conforming radial the
    /// rest of these tests use but with the 2011-2014 era's constant
    /// blocks: no LDM block table, messages straight after the header.
    fn unframed_volume(messages: &[Vec<u8>]) -> Vec<u8> {
        let mut raw = Vec::from(&b"AR2V0006."[..]);
        raw.resize(VOLUME_HEADER_SIZE, 0);
        raw[14..16].copy_from_slice(&20663u16.to_be_bytes());
        raw[16..20].copy_from_slice(&72_196_232u32.to_be_bytes());
        raw[20..24].copy_from_slice(b"KTLX");
        for message in messages {
            raw.extend_from_slice(message);
        }
        raw
    }

    #[test]
    fn the_two_censored_gate_codes_stay_nan_and_stay_told_apart() {
        // Raw 0 is "below threshold" -- the radar looked and detected
        // nothing.  Raw 1 is "range folded" -- second-trip ambiguity that
        // may be a storm.  Both were, and remain, NAN in `data`; the whole
        // point of `censor` is that they are no longer the same NAN.
        let mut builder = RadialBuilder::conforming();
        builder.blocks = vec![
            REAL_VOL_V2.to_vec(),
            elv_block(),
            rad_block(830),
            ref_moment_raw(&[0, 1, 100, 101, 1, 0]),
        ];
        let raw = archive2_volume(&[builder.build()]);
        let file = Level2File::parse_strict(&raw).unwrap();
        let moment = &file.sweeps[0].radials[0].moments[0];

        // The values this decoder has always produced, unchanged.
        assert!(moment.data[0].is_nan());
        assert!(moment.data[1].is_nan());
        assert_eq!(moment.data[2], 17.0);
        assert_eq!(moment.data[3], 17.5);
        assert!(moment.data[4].is_nan());
        assert!(moment.data[5].is_nan());

        // The reason each of them is not a number.
        assert_eq!(
            moment.censor,
            vec![
                censor::BELOW_THRESHOLD,
                censor::RANGE_FOLDED,
                censor::MEASURED,
                censor::MEASURED,
                censor::RANGE_FOLDED,
                censor::BELOW_THRESHOLD,
            ]
        );

        // The two planes describe the same gates, and they agree about
        // which of them are numbers.
        assert_eq!(moment.censor.len(), moment.data.len());
        assert_eq!(moment.censor.len(), moment.gate_count as usize);
        for (value, code) in moment.data.iter().zip(&moment.censor) {
            assert_eq!(value.is_nan(), *code != censor::MEASURED);
        }

        // NOT_COLLECTED is never minted here: this decoder only ever sees
        // gates a radial actually carried.  A consumer that widens a moment
        // into a rectangle owns that code, and the pack builder is the one
        // that does.
        assert!(!moment.censor.contains(&censor::NOT_COLLECTED));
    }

    #[test]
    fn a_16_bit_moment_censors_the_same_two_words_and_no_others() {
        // The 16-bit word path is a separate read in the decode loop, and
        // the sentinels are the same two *values*, not the same two bytes.
        // Raw 2 is a measurement in both widths and must not be censored.
        let mut header = ref_moment(3);
        header.truncate(MOMENT_HEADER_SIZE);
        header[18..20].copy_from_slice(&16u16.to_be_bytes()); // data word size
        header[8..10].copy_from_slice(&3u16.to_be_bytes()); // gate count
        for word in [0u16, 1, 2] {
            header.extend_from_slice(&word.to_be_bytes());
        }
        let mut builder = RadialBuilder::conforming();
        builder.blocks =
            vec![REAL_VOL_V2.to_vec(), elv_block(), rad_block(830), header];
        let raw = archive2_volume(&[builder.build()]);
        let file = Level2File::parse_strict(&raw).unwrap();
        let moment = &file.sweeps[0].radials[0].moments[0];
        assert_eq!(
            moment.censor,
            vec![censor::BELOW_THRESHOLD, censor::RANGE_FOLDED, censor::MEASURED]
        );
        // (2 - 66) / 2
        assert_eq!(moment.data[2], -32.0);
    }

    #[test]
    fn the_framing_discriminator_separates_the_two_real_shapes() {
        // LDM-framed: the first block is a bzip2 stream.
        let framed = archive2_volume(&[RadialBuilder::conforming().build()]);
        assert!(Level2File::is_ldm_framed(&framed));

        // Unframed: the first four bytes after the header are a message's
        // CTM, which in every real pre-2016 volume is zero.
        let unframed = unframed_volume(&[RadialBuilder::conforming().build()]);
        assert!(!Level2File::is_ldm_framed(&unframed));

        // Too short to decide is not LDM: fail closed.
        assert!(!Level2File::is_ldm_framed(&unframed[..VOLUME_HEADER_SIZE]));
        assert!(!Level2File::is_ldm_framed(b"AR2V0006"));
    }

    #[test]
    fn an_uncompressed_message_stream_decodes_under_both_modes() {
        // The pre-2016 `.gz` shape: same radial, same strict checks, no
        // block table.  Carrying the 1.0-era VOL and 20-byte RAD blocks,
        // because that is what those volumes actually pair.
        let mut builder = RadialBuilder::conforming();
        builder.blocks = vec![
            REAL_VOL_V1.to_vec(),
            elv_block(),
            REAL_RAD_V1_ERA.to_vec(),
            ref_moment(4),
        ];
        let raw = unframed_volume(&[builder.build()]);
        for file in [
            Level2File::parse_strict(&raw).unwrap(),
            Level2File::parse(&raw).unwrap(),
        ] {
            assert_eq!(file.station_id, "KTLX");
            assert_eq!(file.sweeps.len(), 1);
            let radial = &file.sweeps[0].radials[0];
            assert_eq!(radial.nyquist_velocity, Some(8.30));
            assert_eq!(radial.moments[0].data, vec![17.0, 17.5, 18.0, 18.5]);
        }

        // Every strict refusal still applies inside the unframed shape: the
        // framing choice is not a relaxation.  A block pointer that leaves
        // its radial is refused here exactly as it is in an LDM volume.
        let mut builder = RadialBuilder::conforming();
        builder.blocks = vec![
            REAL_VOL_V1.to_vec(),
            elv_block(),
            REAL_RAD_V1_ERA.to_vec(),
            ref_moment(4),
        ];
        builder.pointers = Some(vec![48, 92, 104, 40_000]);
        let raw = unframed_volume(&[builder.build()]);
        let err = Level2File::parse_strict(&raw).unwrap_err();
        assert!(err.contains("outside its own"), "{err}");

        // And an unread VOL version is refused in this shape too.
        let mut builder = RadialBuilder::conforming();
        let mut bad_vol = REAL_VOL_V1;
        bad_vol[6] = 4;
        builder.blocks = vec![
            bad_vol.to_vec(),
            elv_block(),
            REAL_RAD_V1_ERA.to_vec(),
            ref_moment(4),
        ];
        let raw = unframed_volume(&[builder.build()]);
        assert!(Level2File::parse_strict(&raw)
            .unwrap_err()
            .contains("version 4.0"));
    }

    #[test]
    fn a_volume_that_yields_no_radial_is_refused_by_strict() {
        // Header only, and header plus a message the decoder steps over:
        // both frame without contradiction and both decode to nothing.  The
        // renderer may draw zero sweeps; a front door may not publish one.
        let mut header_only = Vec::from(&b"AR2V0006."[..]);
        header_only.resize(VOLUME_HEADER_SIZE, 0);
        header_only[20..24].copy_from_slice(b"KTLX");
        let mut skipped = header_only.clone();
        skipped.resize(VOLUME_HEADER_SIZE + 2432, 0);
        skipped[VOLUME_HEADER_SIZE + 15] = 2; // a message type 31 is not

        for raw in [header_only, skipped] {
            let err = Level2File::parse_strict(&raw).unwrap_err();
            assert!(err.contains("no Message-31 radial"), "{err}");
            // Lenient keeps its historical behaviour: an empty volume.
            assert_eq!(Level2File::parse(&raw).unwrap().sweeps.len(), 0);
        }
    }

    #[test]
    fn the_vol_and_elv_gates_run_in_the_strict_volume_walk() {
        // The unit tests above prove the parsers; this proves they are
        // wired, that strict refuses the whole volume, and that lenient --
        // the renderer's mode -- is unchanged by any of it.
        let mut builder = RadialBuilder::conforming();
        let mut bad_vol = REAL_VOL_V2;
        bad_vol[7] = 1; // version 2.1: a layout nothing here has read
        builder.blocks = vec![bad_vol.to_vec(), elv_block(), rad_block(2384), ref_moment(4)];
        let err = strict_error(&builder);
        assert!(err.contains("version 2.1"), "{err}");
        let raw = archive2_volume(&[builder.build()]);
        assert_eq!(Level2File::parse(&raw).unwrap().sweeps.len(), 1);

        let mut builder = RadialBuilder::conforming();
        let mut bad_elv = REAL_ELV;
        bad_elv[4..6].copy_from_slice(&28u16.to_be_bytes());
        builder.blocks = vec![vol_block(), bad_elv.to_vec(), rad_block(2384), ref_moment(4)];
        let err = strict_error(&builder);
        assert!(err.contains("ELV block"), "{err}");
        assert!(err.contains("declares LRTUP 28"), "{err}");

        // The 52-byte version 3.0 block is conforming in the same walk.
        let mut builder = RadialBuilder::conforming();
        builder.blocks = vec![
            REAL_VOL_V3.to_vec(),
            elv_block(),
            rad_block(2384),
            ref_moment(4),
        ];
        let raw = archive2_volume(&[builder.build()]);
        let file = Level2File::parse_strict(&raw).unwrap();
        assert_eq!(file.sweeps[0].radials[0].nyquist_velocity, Some(23.84));
    }

    #[test]
    fn a_sweep_takes_the_smallest_nyquist_its_radials_reported() {
        // The audit's case: 32 m/s first, 20 m/s later.  Taking the first
        // licenses a reported 18 m/s gate under a 25.6 m/s threshold, when
        // that gate's own radial puts the threshold at 16 m/s.
        let mut radials = vec![
            (1, 0, make_radial(0.0, 0.5, 3)),
            (1, 0, make_radial(1.0, 0.5, 1)),
            (1, 0, make_radial(2.0, 0.5, 2)),
        ];
        radials[0].2.nyquist_velocity = Some(32.0);
        radials[1].2.nyquist_velocity = Some(20.0);
        radials[2].2.nyquist_velocity = Some(32.0);
        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(sweeps[0].nyquist_velocity, Some(20.0));
        assert!(sweeps[0].nyquist_radials_disagree);

        // Unanimous radials agree, and say so.
        let mut radials = vec![
            (1, 0, make_radial(0.0, 0.5, 3)),
            (1, 0, make_radial(1.0, 0.5, 2)),
        ];
        for entry in radials.iter_mut() {
            entry.2.nyquist_velocity = Some(23.84);
        }
        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(sweeps[0].nyquist_velocity, Some(23.84));
        assert!(!sweeps[0].nyquist_radials_disagree);

        // A radial that reported nothing is itself a disagreement: the
        // sweep scalar covers a radial that never licensed it.
        let mut radials = vec![
            (1, 0, make_radial(0.0, 0.5, 3)),
            (1, 0, make_radial(1.0, 0.5, 2)),
        ];
        radials[0].2.nyquist_velocity = Some(23.84);
        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(sweeps[0].nyquist_velocity, Some(23.84));
        assert!(sweeps[0].nyquist_radials_disagree);
    }

    #[test]
    fn test_cut_sector_preserved() {
        let radials = vec![
            (1, 3, make_radial(0.0, 0.5, 3)),
            (1, 3, make_radial(1.0, 0.5, 2)),
        ];

        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(sweeps[0].cut_sector, 3);
    }

    #[test]
    fn test_incomplete_sweep_end_status() {
        // Single radial, no end marker — end_status reflects last radial's actual status
        let radials = vec![(1, 0, make_radial(0.0, 0.5, 3))];

        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(sweeps.len(), 1);
        assert_eq!(sweeps[0].start_status, 3);
        // Not 2 or 4, so consumers know this cut didn't end cleanly
        assert_eq!(sweeps[0].end_status, 3);
    }
}
