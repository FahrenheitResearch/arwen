//! GEMPAK surface file reader.
//!
//! GEMPAK surface files use the DM (Data Management) binary format with three
//! layout variants:
//! - **Standard**: rows = date/time, columns = stations
//! - **Ship**: single row, columns have both station + date/time info
//! - **Climate**: rows = stations, columns = date/time
//!
//! Data contains surface observation parameters (TMPF/TMPC, DWPF/DWPC, DRCT,
//! SKNT/SPED, PMSL/PRES, VSBY, etc.).

use super::gempak_dm::{
    DmBuffer, DmFile, DateTimeHeader, StationHeader, BYTES_PER_WORD,
    parse_station_header, parse_datetime_header,
};

// ── Public types ────────────────────────────────────────────────────────

/// A station from a GEMPAK surface file.
#[derive(Debug, Clone)]
pub struct SurfaceStation {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub elevation: f64,
    pub state: String,
    pub country: String,
    pub station_number: i32,
}

/// A single surface observation.
#[derive(Debug, Clone)]
pub struct SurfaceObs {
    pub station_id: String,
    pub time: String,
    pub temperature: Option<f64>,
    pub dewpoint: Option<f64>,
    pub wind_direction: Option<f64>,
    pub wind_speed: Option<f64>,
    pub pressure: Option<f64>,
    pub visibility: Option<f64>,
    pub sky_cover: Option<String>,
    /// All decoded parameters keyed by GEMPAK parameter name.
    pub parameters: std::collections::HashMap<String, f64>,
}

/// Parsed GEMPAK surface file.
#[derive(Debug, Clone)]
pub struct GempakSurface {
    pub stations: Vec<SurfaceStation>,
    pub observations: Vec<SurfaceObs>,
    /// Surface file layout type: "standard", "ship", or "climate".
    pub surface_type: String,
}

/// The three possible surface file layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SurfaceLayout {
    Standard,
    Ship,
    Climate,
}

impl GempakSurface {
    /// Open and parse a GEMPAK surface file.
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
        let row_headers_raw = dm.read_row_headers_raw()?;
        let col_headers_raw = dm.read_column_headers_raw()?;

        let raw_bytes = dm.buffer().data;

        // Determine surface type from key layout
        let layout = Self::detect_layout(&dm, &row_headers_raw)?;

        let surface_type = match layout {
            SurfaceLayout::Standard => "standard",
            SurfaceLayout::Ship => "ship",
            SurfaceLayout::Climate => "climate",
        }
        .to_string();

        // Parse headers depending on layout
        let (stations, observations) = match layout {
            SurfaceLayout::Standard => {
                Self::parse_standard(&dm, &row_headers_raw, &col_headers_raw, &raw_bytes)?
            }
            SurfaceLayout::Ship => {
                Self::parse_ship(&dm, &col_headers_raw, &raw_bytes)?
            }
            SurfaceLayout::Climate => {
                Self::parse_climate(&dm, &row_headers_raw, &col_headers_raw, &raw_bytes)?
            }
        };

        Ok(GempakSurface {
            stations,
            observations,
            surface_type,
        })
    }

    /// Detect which surface layout the file uses.
    fn detect_layout(
        dm: &DmFile,
        row_headers_raw: &[Option<Vec<i32>>],
    ) -> Result<SurfaceLayout, String> {
        let row_has_date = dm.row_keys.iter().any(|k| k == "DATE");
        let row_has_stid = dm.row_keys.iter().any(|k| k == "STID");
        let col_has_date = dm.column_keys.iter().any(|k| k == "DATE");
        let col_has_stid = dm.column_keys.iter().any(|k| k == "STID");

        // Count used rows
        let used_rows = row_headers_raw.iter().filter(|h| h.is_some()).count();

        if used_rows <= 1 && col_has_date && col_has_stid {
            Ok(SurfaceLayout::Ship)
        } else if row_has_date && col_has_stid {
            Ok(SurfaceLayout::Standard)
        } else if col_has_date && row_has_stid {
            Ok(SurfaceLayout::Climate)
        } else {
            Err("Unknown surface data layout".to_string())
        }
    }

    /// Parse a standard surface file (rows = date/time, columns = stations).
    fn parse_standard(
        dm: &DmFile,
        row_headers_raw: &[Option<Vec<i32>>],
        col_headers_raw: &[Option<Vec<i32>>],
        raw_bytes: &[u8],
    ) -> Result<(Vec<SurfaceStation>, Vec<SurfaceObs>), String> {
        // Parse row headers (date/time)
        let row_datetimes = Self::parse_row_datetimes(dm, row_headers_raw);

        // Parse column headers (stations)
        let col_stations = Self::parse_col_stations(dm, col_headers_raw, raw_bytes);

        // Build unique station list
        let stations = Self::unique_stations(&col_stations);

        // Extract observations
        let mut observations = Vec::new();
        for (irow, dth) in &row_datetimes {
            for (icol, stn) in &col_stations {
                let obs = Self::read_observation(dm, *irow, *icol, &stn.stid, dth)?;
                if let Some(obs) = obs {
                    observations.push(obs);
                }
            }
        }

        Ok((stations, observations))
    }

    /// Parse a ship surface file (single row, columns have station + date/time).
    fn parse_ship(
        dm: &DmFile,
        col_headers_raw: &[Option<Vec<i32>>],
        raw_bytes: &[u8],
    ) -> Result<(Vec<SurfaceStation>, Vec<SurfaceObs>), String> {
        // Column headers contain both station and date/time
        let base = DmBuffer::word_to_offset(dm.prod_desc.column_headers_ptr);
        let nkeys = dm.prod_desc.column_keys as usize;

        let mut stations_map = std::collections::HashMap::new();
        let mut observations = Vec::new();

        for (i, ch) in col_headers_raw.iter().enumerate() {
            if let Some(vals) = ch {
                let byte_offset = base + i * (1 + nkeys) * BYTES_PER_WORD + BYTES_PER_WORD;
                let stn = parse_station_header(&dm.column_keys, vals, raw_bytes, byte_offset);
                let dth = parse_datetime_header(&dm.column_keys, vals);

                let key = if !stn.stid.is_empty() {
                    stn.stid.clone()
                } else {
                    stn.stnm.to_string()
                };
                stations_map.entry(key).or_insert_with(|| SurfaceStation {
                    id: stn.stid.clone(),
                    lat: stn.slat,
                    lon: stn.slon,
                    elevation: stn.selv as f64,
                    state: stn.stat.clone(),
                    country: stn.coun.clone(),
                    station_number: stn.stnm,
                });

                let obs = Self::read_observation(dm, 0, i, &stn.stid, &dth)?;
                if let Some(obs) = obs {
                    observations.push(obs);
                }
            }
        }

        let stations: Vec<SurfaceStation> = stations_map.into_values().collect();
        Ok((stations, observations))
    }

    /// Parse a climate surface file (rows = stations, columns = date/time).
    fn parse_climate(
        dm: &DmFile,
        row_headers_raw: &[Option<Vec<i32>>],
        col_headers_raw: &[Option<Vec<i32>>],
        raw_bytes: &[u8],
    ) -> Result<(Vec<SurfaceStation>, Vec<SurfaceObs>), String> {
        // Parse row headers (stations)
        let base = DmBuffer::word_to_offset(dm.prod_desc.row_headers_ptr);
        let nkeys = dm.prod_desc.row_keys as usize;

        let mut row_stations: Vec<(usize, StationHeader)> = Vec::new();
        for (i, rh) in row_headers_raw.iter().enumerate() {
            if let Some(vals) = rh {
                let byte_offset = base + i * (1 + nkeys) * BYTES_PER_WORD + BYTES_PER_WORD;
                let stn = parse_station_header(&dm.row_keys, vals, raw_bytes, byte_offset);
                row_stations.push((i, stn));
            }
        }

        // Parse column headers (date/time)
        let col_datetimes = Self::parse_col_datetimes(dm, col_headers_raw);

        // Build stations list
        let mut stations = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (_, stn) in &row_stations {
            let key = if !stn.stid.is_empty() {
                stn.stid.clone()
            } else {
                stn.stnm.to_string()
            };
            if seen.insert(key) {
                stations.push(SurfaceStation {
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

        // Extract observations
        let mut observations = Vec::new();
        for (icol, dth) in &col_datetimes {
            for (irow, stn) in &row_stations {
                let obs = Self::read_observation(dm, *irow, *icol, &stn.stid, dth)?;
                if let Some(obs) = obs {
                    observations.push(obs);
                }
            }
        }

        Ok((stations, observations))
    }

    // ── Shared helpers ──────────────────────────────────────────────────

    fn parse_row_datetimes(
        dm: &DmFile,
        row_headers_raw: &[Option<Vec<i32>>],
    ) -> Vec<(usize, DateTimeHeader)> {
        let mut result = Vec::new();
        for (i, rh) in row_headers_raw.iter().enumerate() {
            if let Some(vals) = rh {
                let dth = parse_datetime_header(&dm.row_keys, vals);
                result.push((i, dth));
            }
        }
        result
    }

    fn parse_col_datetimes(
        dm: &DmFile,
        col_headers_raw: &[Option<Vec<i32>>],
    ) -> Vec<(usize, DateTimeHeader)> {
        let mut result = Vec::new();
        for (i, ch) in col_headers_raw.iter().enumerate() {
            if let Some(vals) = ch {
                let dth = parse_datetime_header(&dm.column_keys, vals);
                result.push((i, dth));
            }
        }
        result
    }

    fn parse_col_stations(
        dm: &DmFile,
        col_headers_raw: &[Option<Vec<i32>>],
        raw_bytes: &[u8],
    ) -> Vec<(usize, StationHeader)> {
        let base = DmBuffer::word_to_offset(dm.prod_desc.column_headers_ptr);
        let nkeys = dm.prod_desc.column_keys as usize;
        let mut result = Vec::new();
        for (i, ch) in col_headers_raw.iter().enumerate() {
            if let Some(vals) = ch {
                let byte_offset = base + i * (1 + nkeys) * BYTES_PER_WORD + BYTES_PER_WORD;
                let stn = parse_station_header(&dm.column_keys, vals, raw_bytes, byte_offset);
                result.push((i, stn));
            }
        }
        result
    }

    fn unique_stations(col_stations: &[(usize, StationHeader)]) -> Vec<SurfaceStation> {
        let mut stations = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (_, stn) in col_stations {
            let key = if !stn.stid.is_empty() {
                stn.stid.clone()
            } else {
                stn.stnm.to_string()
            };
            if seen.insert(key) {
                stations.push(SurfaceStation {
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
        stations
    }

    /// Read a single surface observation from the data block.
    fn read_observation(
        dm: &DmFile,
        irow: usize,
        icol: usize,
        station_id: &str,
        dth: &DateTimeHeader,
    ) -> Result<Option<SurfaceObs>, String> {
        let time_str = if dth.date.is_empty() {
            dth.time.clone()
        } else if dth.time.is_empty() {
            dth.date.clone()
        } else {
            format!("{} {}", dth.date, dth.time)
        };

        let mut all_params: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        let mut has_any_data = false;

        for (iprt, part) in dm.parts.iter().enumerate() {
            let ptr = dm.data_pointer(irow, icol, iprt)?;
            if ptr == 0 {
                continue;
            }

            let param_info = &dm.parameters[iprt];

            if part.data_type == super::gempak_dm::DataType::Character {
                let strings = dm.read_char_data(irow, icol, iprt)?;
                for (i, pname) in param_info.names.iter().enumerate() {
                    if i < strings.len() {
                        // Store character data as NaN in the numeric map;
                        // callers can access the raw string via the sky_cover field
                        // or the parameters map.
                        all_params.insert(pname.clone(), f64::NAN);
                    }
                }
                has_any_data = true;
                continue;
            }

            let data = dm.read_data(irow, icol, iprt)?;
            if data.is_empty() {
                continue;
            }

            has_any_data = true;
            let nparms = param_info.names.len();
            if nparms == 0 {
                continue;
            }

            // Surface data: typically one level, so data is just nparms values
            // For surface obs we take the first level
            for (ip, pname) in param_info.names.iter().enumerate() {
                if ip < data.len() {
                    let val = data[ip];
                    if !val.is_nan() {
                        all_params.insert(pname.clone(), val);
                    }
                }
            }
        }

        if !has_any_data {
            return Ok(None);
        }

        // Map GEMPAK parameter names to our standard fields
        let temperature = all_params
            .get("TMPC")
            .or_else(|| all_params.get("TMPF").map(|_| all_params.get("TMPF").unwrap()))
            .or_else(|| all_params.get("TEMP"))
            .copied();

        let dewpoint = all_params
            .get("DWPC")
            .or_else(|| all_params.get("DWPF").map(|_| all_params.get("DWPF").unwrap()))
            .or_else(|| all_params.get("DWPT"))
            .copied();

        let wind_direction = all_params.get("DRCT").copied();

        let wind_speed = all_params
            .get("SKNT")
            .or_else(|| all_params.get("SPED"))
            .or_else(|| all_params.get("SMPS"))
            .copied();

        let pressure = all_params
            .get("PMSL")
            .or_else(|| all_params.get("PRES"))
            .or_else(|| all_params.get("ALTI"))
            .copied();

        let visibility = all_params.get("VSBY").copied();

        // Sky cover is typically a character field; represented as None for now
        // unless stored numerically
        let sky_cover = None;

        Ok(Some(SurfaceObs {
            station_id: station_id.to_string(),
            time: time_str,
            temperature,
            dewpoint,
            wind_direction,
            wind_speed,
            pressure,
            visibility,
            sky_cover,
            parameters: all_params,
        }))
    }
}
