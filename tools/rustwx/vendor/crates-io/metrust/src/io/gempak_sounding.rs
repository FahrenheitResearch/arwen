//! GEMPAK sounding file reader.
//!
//! GEMPAK sounding files use the DM (Data Management) binary format with:
//! - Rows = date/times
//! - Columns = stations
//! - Data = vertical profile parameters (PRES, TEMP, DWPT, DRCT, SPED, HGHT, ...)
//!
//! Supports both merged (SNDT part) and unmerged (TTAA/TTBB/PPAA/PPBB parts)
//! sounding formats. Unmerged data is read part-by-part.

use super::gempak_dm::{
    DmBuffer, DmFile, DateTimeHeader, StationHeader, BYTES_PER_WORD,
    parse_station_header, parse_datetime_header,
};

// ── Public types ────────────────────────────────────────────────────────

/// A station entry from a GEMPAK sounding file.
#[derive(Debug, Clone)]
pub struct GempakStation {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub elevation: f64,
    pub state: String,
    pub country: String,
    pub station_number: i32,
}

/// A single vertical sounding profile.
#[derive(Debug, Clone)]
pub struct SoundingData {
    pub station_id: String,
    pub time: String,
    pub pressure: Vec<f64>,
    pub temperature: Vec<f64>,
    pub dewpoint: Vec<f64>,
    pub wind_direction: Vec<f64>,
    pub wind_speed: Vec<f64>,
    pub height: Vec<f64>,
    /// All decoded parameters keyed by GEMPAK parameter name.
    /// Each value vector has one element per vertical level.
    pub parameters: std::collections::HashMap<String, Vec<f64>>,
}

/// Parsed GEMPAK sounding file.
#[derive(Debug, Clone)]
pub struct GempakSounding {
    pub stations: Vec<GempakStation>,
    pub soundings: Vec<SoundingData>,
    /// Whether the file uses merged (SNDT) format.
    pub merged: bool,
}

impl GempakSounding {
    /// Open and parse a GEMPAK sounding file.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let dm = DmFile::from_file(path)?;
        Self::from_dm(dm)
    }

    /// Parse from raw bytes.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        let dm = DmFile::from_bytes(data)?;
        Self::from_dm(dm)
    }

    fn from_dm(dm: DmFile) -> Result<Self, String> {
        // Determine if merged format (has SNDT part)
        let merged = dm.parts.iter().any(|p| p.name == "SNDT");

        // Read row headers (date/time) and column headers (stations)
        let row_headers_raw = dm.read_row_headers_raw()?;
        let col_headers_raw = dm.read_column_headers_raw()?;

        let raw_bytes = dm.buffer().data;

        // Parse row headers (date/time info)
        let mut row_datetimes: Vec<(usize, DateTimeHeader)> = Vec::new();
        {
            let base = DmBuffer::word_to_offset(dm.prod_desc.row_headers_ptr);
            let nkeys = dm.prod_desc.row_keys as usize;
            for (i, rh) in row_headers_raw.iter().enumerate() {
                if let Some(vals) = rh {
                    // +1 word for the USED_FLAG
                    let _byte_offset = base + i * (1 + nkeys) * BYTES_PER_WORD + BYTES_PER_WORD;
                    let dth = parse_datetime_header(&dm.row_keys, vals);
                    row_datetimes.push((i, dth));
                }
            }
        }

        // Parse column headers (station info)
        let mut col_stations: Vec<(usize, StationHeader)> = Vec::new();
        {
            let base = DmBuffer::word_to_offset(dm.prod_desc.column_headers_ptr);
            let nkeys = dm.prod_desc.column_keys as usize;
            for (i, ch) in col_headers_raw.iter().enumerate() {
                if let Some(vals) = ch {
                    let byte_offset = base + i * (1 + nkeys) * BYTES_PER_WORD + BYTES_PER_WORD;
                    let stn = parse_station_header(&dm.column_keys, vals, &raw_bytes, byte_offset);
                    col_stations.push((i, stn));
                }
            }
        }

        // Build unique station list
        let mut stations: Vec<GempakStation> = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        for (_, stn) in &col_stations {
            let key = if !stn.stid.is_empty() {
                stn.stid.clone()
            } else {
                stn.stnm.to_string()
            };
            if seen_ids.insert(key) {
                stations.push(GempakStation {
                    id: stn.stid.clone(),
                    lat: stn.slat,
                    lon: stn.slon,
                    elevation: stn.selv as f64,
                    state: stn.stat.clone(),
                    country: stn.coun.clone(),
                    station_number: stn.stnm,
                });
            }
        }

        // Find all (row, col) pairs that have data and extract soundings
        let mut soundings: Vec<SoundingData> = Vec::new();

        for (irow, dth) in &row_datetimes {
            for (icol, stn) in &col_stations {
                // Check if any part has data for this row/col
                let mut has_data = false;
                for iprt in 0..dm.parts.len() {
                    let ptr = dm.data_pointer(*irow, *icol, iprt)?;
                    if ptr != 0 {
                        has_data = true;
                        break;
                    }
                }

                if !has_data {
                    continue;
                }

                // Build datetime string
                let time_str = if dth.date.is_empty() {
                    dth.time.clone()
                } else if dth.time.is_empty() {
                    dth.date.clone()
                } else {
                    format!("{} {}", dth.date, dth.time)
                };

                // Read all parts and merge parameters
                let mut all_params: std::collections::HashMap<String, Vec<f64>> =
                    std::collections::HashMap::new();

                for (iprt, _part) in dm.parts.iter().enumerate() {
                    let data = dm.read_data(*irow, *icol, iprt)?;
                    if data.is_empty() {
                        continue;
                    }

                    let param_info = &dm.parameters[iprt];
                    let nparms = param_info.names.len();
                    if nparms == 0 {
                        continue;
                    }

                    // Data is interleaved: val0_p0, val0_p1, ..., val1_p0, val1_p1, ...
                    let nlevels = data.len() / nparms;
                    for (ip, pname) in param_info.names.iter().enumerate() {
                        let mut col_data = Vec::with_capacity(nlevels);
                        for ilev in 0..nlevels {
                            col_data.push(data[ilev * nparms + ip]);
                        }
                        all_params.insert(pname.clone(), col_data);
                    }
                }

                // Extract standard sounding vectors, falling back to empty
                let nlevels = all_params.values().map(|v| v.len()).max().unwrap_or(0);
                let empty = vec![f64::NAN; nlevels];

                let pressure = all_params.get("PRES").cloned().unwrap_or_else(|| empty.clone());
                let temperature = all_params.get("TEMP").cloned().unwrap_or_else(|| empty.clone());
                let dewpoint = all_params.get("DWPT").cloned().unwrap_or_else(|| empty.clone());
                let wind_direction = all_params.get("DRCT").cloned().unwrap_or_else(|| empty.clone());
                let wind_speed = all_params.get("SPED").cloned()
                    .or_else(|| all_params.get("SKNT").cloned())
                    .unwrap_or_else(|| empty.clone());
                let height = all_params.get("HGHT").cloned().unwrap_or_else(|| empty.clone());

                soundings.push(SoundingData {
                    station_id: stn.stid.clone(),
                    time: time_str,
                    pressure,
                    temperature,
                    dewpoint,
                    wind_direction,
                    wind_speed,
                    height,
                    parameters: all_params,
                });
            }
        }

        Ok(GempakSounding {
            stations,
            soundings,
            merged,
        })
    }
}
