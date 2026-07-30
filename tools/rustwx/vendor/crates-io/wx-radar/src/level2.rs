//! NEXRAD Level-II (Archive II) file parser.
//! Specification: ICD 2620010H (RDA/RPG).

use byteorder::{BigEndian, ReadBytesExt};
use bzip2::read::BzDecoder;
use rayon::prelude::*;
use std::io::{Cursor, Read};
use chrono::{Datelike, NaiveDate};

use crate::products::RadarProduct;

const VOLUME_HEADER_SIZE: usize = 24;
const MSG_HEADER_SIZE: usize = 16;

#[derive(Debug, Clone)]
pub struct Level2File {
    pub station_id: String,
    pub volume_date: u16,
    pub volume_time: u32,
    pub sweeps: Vec<Level2Sweep>,
}

#[derive(Debug, Clone)]
pub struct Level2Sweep {
    pub elevation_number: u8,
    pub elevation_angle: f32,
    pub nyquist_velocity: Option<f32>,
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

#[derive(Debug, Clone)]
pub struct MomentData {
    pub product: RadarProduct,
    pub gate_count: u16,
    pub first_gate_range: u16,
    pub gate_size: u16,
    pub data: Vec<f32>,
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
}

impl Level2File {
    pub fn parse(raw_data: &[u8]) -> Result<Self, String> {
        let header_str = String::from_utf8_lossy(&raw_data[..raw_data.len().min(9)]);

        let data = if header_str.starts_with("AR2V") || header_str.starts_with("ARCH") {
            Self::decompress_archive2(raw_data)?
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

        while (cursor.position() as usize) < data.len().saturating_sub(MSG_HEADER_SIZE) {
            match Self::read_message(&mut cursor, &data) {
                Ok(Some((elev_num, cut_sector, radial))) => {
                    all_radials.push((elev_num, cut_sector, radial));
                }
                Ok(None) => continue,
                Err(_) => break,
            }
        }

        let sweeps = Self::split_radials_into_sweeps(all_radials);

        Ok(Level2File {
            station_id: header.station_id,
            volume_date: header.volume_date,
            volume_time: header.volume_time,
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
            date.year(), date.month(), date.day(), hours, minutes, seconds,
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

    fn decompress_archive2(raw_data: &[u8]) -> Result<Vec<u8>, String> {
        if raw_data.len() < VOLUME_HEADER_SIZE {
            return Err("Data too short for volume header".into());
        }

        let mut blocks: Vec<(usize, usize, bool)> = Vec::new();
        let mut pos = VOLUME_HEADER_SIZE;

        while pos + 4 <= raw_data.len() {
            let block_size = i32::from_be_bytes([
                raw_data[pos], raw_data[pos + 1], raw_data[pos + 2], raw_data[pos + 3],
            ]);
            pos += 4;
            let actual_size = block_size.unsigned_abs() as usize;
            if pos + actual_size > raw_data.len() { break; }
            let is_bz2 = actual_size >= 2 && raw_data[pos] == b'B' && raw_data[pos + 1] == b'Z';
            blocks.push((pos, actual_size, is_bz2));
            pos += actual_size;
        }

        let decompressed: Vec<Vec<u8>> = blocks.par_iter()
            .map(|&(start, len, is_bz2)| {
                let block_data = &raw_data[start..start + len];
                if is_bz2 {
                    let mut decoder = BzDecoder::new(block_data);
                    let mut out = Vec::new();
                    match decoder.read_to_end(&mut out) {
                        Ok(_) => out,
                        Err(_) => block_data.to_vec(),
                    }
                } else {
                    block_data.to_vec()
                }
            })
            .collect();

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

        Ok(VolumeHeader { station_id, volume_date, volume_time })
    }

    /// Returns (elevation_number, cut_sector, radial).
    fn read_message(cursor: &mut Cursor<&Vec<u8>>, data: &[u8]) -> Result<Option<(u8, u8, RadialData)>, String> {
        let start_pos = cursor.position() as usize;
        if start_pos + 12 > data.len() { return Err("End of data".into()); }

        let mut ctm = [0u8; 12];
        cursor.read_exact(&mut ctm).map_err(|e| e.to_string())?;

        if (cursor.position() as usize) + MSG_HEADER_SIZE > data.len() {
            return Err("End of data".into());
        }

        let msg_header = Self::read_message_header(cursor)?;

        if msg_header.message_type != 31 {
            let next_pos = start_pos + 2432;
            if next_pos <= data.len() {
                cursor.set_position(next_pos as u64);
            } else {
                return Err("End of data".into());
            }
            return Ok(None);
        }

        let msg31_start = cursor.position() as usize;
        let msg31 = Self::read_msg31_header(cursor)?;

        let mut block_pointers = Vec::new();
        for _ in 0..msg31.data_block_count {
            let offset = cursor.read_u32::<BigEndian>().map_err(|e| e.to_string())?;
            block_pointers.push(offset);
        }

        let mut moments = Vec::new();
        let mut nyquist_velocity: Option<f32> = None;

        for ptr_offset in &block_pointers {
            let block_pos = msg31_start + *ptr_offset as usize;
            if block_pos + 4 > data.len() { continue; }

            let block_type = data[block_pos];
            if block_type == b'D' {
                if let Ok(moment) = Self::parse_moment_block(data, block_pos) {
                    // Skip unknown/unrecognized moment types
                    if moment.product != RadarProduct::Unknown {
                        moments.push(moment);
                    }
                }
            } else if block_type == b'R' {
                nyquist_velocity = Self::parse_radial_block_nyquist(data, block_pos);
            }
        }

        let msg_size_bytes = (msg_header.message_size as usize) * 2 + 12;
        let next_pos = start_pos + msg_size_bytes.max(2432);
        if next_pos <= data.len() {
            cursor.set_position(next_pos as u64);
        }

        let radial = RadialData {
            azimuth: msg31.azimuth_angle,
            elevation: msg31.elevation_angle,
            azimuth_spacing: if msg31.azimuth_resolution == 1 { 0.5 } else { 1.0 },
            nyquist_velocity,
            radial_status: msg31.radial_status,
            moments,
        };

        Ok(Some((msg31.elevation_number, msg31.cut_sector, radial)))
    }

    fn read_message_header(cursor: &mut Cursor<&Vec<u8>>) -> Result<MessageHeader, String> {
        let message_size = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let _rda_channel = cursor.read_u8().map_err(|e| e.to_string())?;
        let message_type = cursor.read_u8().map_err(|e| e.to_string())?;
        // Skip remaining 12 bytes of header
        let mut skip = [0u8; 12];
        cursor.read_exact(&mut skip).map_err(|e| e.to_string())?;
        Ok(MessageHeader { message_size, message_type })
    }

    fn read_msg31_header(cursor: &mut Cursor<&Vec<u8>>) -> Result<Message31Header, String> {
        let mut radar_id = [0u8; 4];
        cursor.read_exact(&mut radar_id).map_err(|e| e.to_string())?;
        let _collection_time = cursor.read_u32::<BigEndian>().map_err(|e| e.to_string())?;
        let _collection_date = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let _azimuth_number = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let azimuth_angle = cursor.read_f32::<BigEndian>().map_err(|e| e.to_string())?;
        let _compression = cursor.read_u8().map_err(|e| e.to_string())?;
        let _spare = cursor.read_u8().map_err(|e| e.to_string())?;
        let _radial_length = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;
        let azimuth_resolution = cursor.read_u8().map_err(|e| e.to_string())?;
        let radial_status = cursor.read_u8().map_err(|e| e.to_string())?;
        let elevation_number = cursor.read_u8().map_err(|e| e.to_string())?;
        let cut_sector = cursor.read_u8().map_err(|e| e.to_string())?;
        let elevation_angle = cursor.read_f32::<BigEndian>().map_err(|e| e.to_string())?;
        let _spot_blanking = cursor.read_u8().map_err(|e| e.to_string())?;
        let _az_index_mode = cursor.read_u8().map_err(|e| e.to_string())?;
        let data_block_count = cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())?;

        Ok(Message31Header {
            azimuth_angle, elevation_angle, elevation_number,
            azimuth_resolution, radial_status, cut_sector, data_block_count,
        })
    }

    fn parse_radial_block_nyquist(data: &[u8], offset: usize) -> Option<f32> {
        if offset + 28 > data.len() { return None; }
        let nyquist_raw = u16::from_be_bytes([data[offset + 26], data[offset + 27]]);
        if nyquist_raw == 0 { return None; }
        Some(nyquist_raw as f32 / 100.0)
    }

    fn parse_moment_block(data: &[u8], offset: usize) -> Result<MomentData, String> {
        if offset + 28 > data.len() { return Err("Moment block too short".into()); }

        let mut cursor = Cursor::new(&data[offset..]);
        let _block_type = cursor.read_u8().map_err(|e| e.to_string())?;
        let mut name_bytes = [0u8; 3];
        cursor.read_exact(&mut name_bytes).map_err(|e| e.to_string())?;
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
        for _ in 0..gate_count {
            let raw = if data_word_size >= 16 {
                cursor.read_u16::<BigEndian>().map_err(|e| e.to_string())? as u32
            } else {
                cursor.read_u8().map_err(|e| e.to_string())? as u32
            };
            let value = if raw <= 1 {
                f32::NAN
            } else {
                (raw as f32 - offset_val) / scale
            };
            decoded.push(value);
        }

        Ok(MomentData { product, gate_count, first_gate_range, gate_size, data: decoded })
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
            if radials.is_empty() { return; }
            let elev_angle = radials[0].elevation;
            let nyquist = radials.iter().find_map(|r| r.nyquist_velocity);
            let start_status = radials[0].radial_status;
            let end_status = radials.last().map(|r| r.radial_status).unwrap_or(0xFF);
            sweeps.push(Level2Sweep {
                elevation_number: elev_num,
                elevation_angle: elev_angle,
                nyquist_velocity: nyquist,
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
                    &mut sweeps, &mut current_radials,
                    current_elev_num, current_cut_sector, &mut sweep_counter,
                );
            }

            current_elev_num = elev_num;
            current_cut_sector = cut_sector;
            current_radials.push(radial);
        }

        flush(
            &mut sweeps, &mut current_radials,
            current_elev_num, current_cut_sector, &mut sweep_counter,
        );

        sweeps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            (1, 0, make_radial(0.0, 0.5, 3)),   // start volume
            (1, 0, make_radial(1.0, 0.5, 1)),
            (1, 0, make_radial(2.0, 0.5, 2)),   // end elev
            // Tilt 2: elev_num=2, 0.9°
            (2, 0, make_radial(0.0, 0.9, 0)),   // start elev
            (2, 0, make_radial(1.0, 0.9, 1)),
            (2, 0, make_radial(2.0, 0.9, 2)),   // end elev
            // Tilt 3: elev_num=3, 1.3°
            (3, 0, make_radial(0.0, 1.3, 0)),   // start elev
            (3, 0, make_radial(1.0, 1.3, 1)),
            (3, 0, make_radial(2.0, 1.3, 4)),   // end volume
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
        assert_eq!(sweeps.len(), 4, "SAILS repeat should produce 4 sweeps, not 3");
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
            (2, 0, make_radial(0.0, 0.9, 1)),  // elev change but no status marker
            (2, 0, make_radial(1.0, 0.9, 1)),
        ];

        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(sweeps.len(), 2, "elevation_number change should split even without status marker");
        assert_eq!(sweeps[0].elevation_number, 1);
        assert_eq!(sweeps[1].elevation_number, 2);
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
        let radials = vec![
            (1, 0, make_radial(0.0, 0.5, 3)),
        ];

        let sweeps = Level2File::split_radials_into_sweeps(radials);
        assert_eq!(sweeps.len(), 1);
        assert_eq!(sweeps[0].start_status, 3);
        // Not 2 or 4, so consumers know this cut didn't end cleanly
        assert_eq!(sweeps[0].end_status, 3);
    }
}
