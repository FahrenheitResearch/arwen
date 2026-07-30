use chrono::NaiveDateTime;

/// A parsed GRIB2 file containing one or more messages.
#[derive(Debug, Clone)]
pub struct Grib2File {
    pub messages: Vec<Grib2Message>,
}

/// A single GRIB2 message (one field/variable).
#[derive(Debug, Clone)]
pub struct Grib2Message {
    pub discipline: u8,
    pub identification: Identification,
    pub reference_time: NaiveDateTime,
    pub grid: GridDefinition,
    pub product: ProductDefinition,
    pub data_rep: DataRepresentation,
    pub bitmap: Option<Vec<bool>>,
    pub raw_data: Vec<u8>,
}

/// Origin and production identity from GRIB2 Section 1.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct Identification {
    pub center_id: u16,
    pub subcenter_id: u16,
    pub master_table_version: u8,
    pub local_table_version: u8,
    pub reference_time_significance: u8,
    pub production_status: u8,
    pub processed_data_type: u8,
}

/// Grid definition from Section 3.
#[derive(Debug, Clone, PartialEq)]
pub struct GridDefinition {
    pub template: u16,
    pub nx: u32,
    pub ny: u32,
    pub lat1: f64,
    pub lon1: f64,
    pub lat2: f64,
    pub lon2: f64,
    pub dx: f64,
    pub dy: f64,
    pub latin1: f64,
    pub latin2: f64,
    pub lov: f64,
    pub scan_mode: u8,
    /// Latitude where Dx and Dy are specified (used by Polar Stereographic, Mercator).
    pub lad: f64,
    /// Projection center flag: 0 = North Pole on projection plane,
    /// 1 = South Pole on projection plane (Polar Stereographic).
    pub projection_center_flag: u8,
    /// Number of parallels between a pole and the equator (Gaussian grids).
    pub n_parallel: u32,
    /// Rotated grid: latitude of the southern pole of rotation (degrees).
    pub south_pole_lat: f64,
    /// Rotated grid: longitude of the southern pole of rotation (degrees).
    pub south_pole_lon: f64,
    /// Rotated grid: angle of rotation (degrees).
    pub rotation_angle: f64,
    /// Space view: sub-satellite point latitude (degrees).
    pub satellite_lat: f64,
    /// Space view: sub-satellite point longitude (degrees).
    pub satellite_lon: f64,
    /// Space view: apparent diameter of Earth in grid lengths, x-direction.
    pub xp: f64,
    /// Space view: apparent diameter of Earth in grid lengths, y-direction.
    pub yp: f64,
    /// Space view: altitude of the camera above the Earth's surface (m).
    pub altitude: f64,
    /// Points per latitude row for reduced Gaussian grids (from pl array).
    pub pl: Option<Vec<u32>>,
    /// Whether this is a reduced (quasi-regular) grid.
    pub is_reduced: bool,
    /// Actual number of data points from section 3 octets 7-10.
    pub num_data_points: u32,
    /// Shape of the Earth (Code Table 3.2): 0=sphere, 6=WGS84, etc.
    pub shape_of_earth: u8,
    /// Resolution and component flags byte.
    pub resolution_flags: u8,
}

/// Product definition from Section 4.
#[derive(Debug, Clone)]
pub struct ProductDefinition {
    pub template: u16,
    pub parameter_category: u8,
    pub parameter_number: u8,
    pub generating_process: u8,
    pub background_generating_process_id: u8,
    pub forecast_generating_process_id: u8,
    pub forecast_time: u32,
    pub time_range_unit: u8,
    pub level_type: u8,
    pub level_value: f64,
    /// Type of the second fixed surface (Code Table 4.5), or 255 when absent.
    pub second_level_type: u8,
    /// Scaled value of the second fixed surface.
    pub second_level_value: f64,
    /// Type of ensemble forecast (PDT 4.1, 4.11).
    pub ensemble_type: Option<u8>,
    /// Perturbation number — the "number" key (PDT 4.1, 4.11).
    pub perturbation_number: Option<u8>,
    /// Number of forecasts in ensemble — the "totalNumber" key (PDT 4.1, 4.2, 4.11, 4.12).
    pub num_forecasts_in_ensemble: Option<u8>,
    /// Derived forecast type (PDT 4.2, 4.12).
    pub derived_forecast_type: Option<u8>,
    /// Percentile value (PDT 4.6, 4.10).
    pub percentile_value: Option<u8>,
    /// Forecast probability number (PDT 4.5, 4.9).
    pub probability_number: Option<u8>,
    /// Total number of forecast probabilities (PDT 4.5, 4.9).
    pub total_number_of_probabilities: Option<u8>,
    /// Probability type (PDT 4.5, 4.9; Code Table 4.9).
    pub probability_type: Option<u8>,
    /// Lower probability limit after applying its decimal scale (PDT 4.5, 4.9).
    pub probability_lower_limit: Option<f64>,
    /// Upper probability limit after applying its decimal scale (PDT 4.5, 4.9).
    pub probability_upper_limit: Option<f64>,
    /// Type of statistical processing (PDT 4.8, 4.11, 4.12).
    pub statistical_process_type: Option<u8>,
    /// End of overall time interval (PDT 4.8, 4.11, 4.12).
    pub end_of_interval: Option<NaiveDateTime>,
    /// Indicator of unit of time range for the first statistical time-range specification
    /// (PDT 4.8, 4.11, 4.12).
    pub statistical_time_range_unit: Option<u8>,
    /// Length of statistical time range (PDT 4.8, 4.11, 4.12).
    pub time_range_length: Option<u32>,
}

/// Data representation from Section 5.
#[derive(Debug, Clone)]
pub struct DataRepresentation {
    pub template: u16,
    pub reference_value: f32,
    pub binary_scale: i16,
    pub decimal_scale: i16,
    pub bits_per_value: u8,
    /// Type of original field values (Code Table 5.1).
    pub original_field_type: u8,
    pub group_splitting_method: u8,
    pub num_groups: u32,
    pub group_width_ref: u8,
    pub group_width_bits: u8,
    pub group_length_ref: u32,
    pub group_length_inc: u8,
    pub last_group_length: u32,
    pub group_length_bits: u8,
    pub spatial_diff_order: u8,
    pub spatial_diff_bytes: u8,
    /// CCSDS (Template 5.42): compression flags.
    pub ccsds_flags: u16,
    /// CCSDS (Template 5.42): block size.
    pub ccsds_block_size: u16,
    /// CCSDS (Template 5.42): reference sample interval.
    pub ccsds_rsi: u16,
    /// Number of data points from Section 5 (bytes 6-9).
    pub section5_num_data_points: u32,
}

impl Default for GridDefinition {
    fn default() -> Self {
        Self {
            template: 0,
            nx: 0,
            ny: 0,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 0.0,
            lon2: 0.0,
            dx: 0.0,
            dy: 0.0,
            latin1: 0.0,
            latin2: 0.0,
            lov: 0.0,
            scan_mode: 0,
            lad: 0.0,
            projection_center_flag: 0,
            n_parallel: 0,
            south_pole_lat: 0.0,
            south_pole_lon: 0.0,
            rotation_angle: 0.0,
            satellite_lat: 0.0,
            satellite_lon: 0.0,
            xp: 0.0,
            yp: 0.0,
            altitude: 0.0,
            pl: None,
            is_reduced: false,
            num_data_points: 0,
            shape_of_earth: 0,
            resolution_flags: 0,
        }
    }
}

impl ProductDefinition {
    /// Returns the first statistical time range expressed in hours when PDT 4.8/4.11/4.12
    /// provides an hourly window. Falls back to the forecast-time unit for callers that only
    /// populated `time_range_length`.
    pub fn statistical_time_range_hours(&self) -> Option<u16> {
        let unit = self
            .statistical_time_range_unit
            .unwrap_or(self.time_range_unit);
        if unit != 1 {
            return None;
        }
        self.time_range_length
            .and_then(|hours| u16::try_from(hours).ok())
    }
}

impl Default for ProductDefinition {
    fn default() -> Self {
        Self {
            template: 0,
            parameter_category: 0,
            parameter_number: 0,
            generating_process: 0,
            background_generating_process_id: 0,
            forecast_generating_process_id: 0,
            forecast_time: 0,
            time_range_unit: 0,
            level_type: 0,
            level_value: 0.0,
            second_level_type: 255,
            second_level_value: 0.0,
            ensemble_type: None,
            perturbation_number: None,
            num_forecasts_in_ensemble: None,
            derived_forecast_type: None,
            percentile_value: None,
            probability_number: None,
            total_number_of_probabilities: None,
            probability_type: None,
            probability_lower_limit: None,
            probability_upper_limit: None,
            statistical_process_type: None,
            end_of_interval: None,
            statistical_time_range_unit: None,
            time_range_length: None,
        }
    }
}

impl Default for DataRepresentation {
    fn default() -> Self {
        Self {
            template: 0,
            reference_value: 0.0,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 0,
            original_field_type: 0,
            group_splitting_method: 0,
            num_groups: 0,
            group_width_ref: 0,
            group_width_bits: 0,
            group_length_ref: 0,
            group_length_inc: 0,
            last_group_length: 0,
            group_length_bits: 0,
            spatial_diff_order: 0,
            spatial_diff_bytes: 0,
            ccsds_flags: 0,
            ccsds_block_size: 0,
            ccsds_rsi: 0,
            section5_num_data_points: 0,
        }
    }
}

// ---------- helper readers ----------

fn read_u8(data: &[u8], offset: usize) -> Result<u8, String> {
    data.get(offset).copied().ok_or_else(|| {
        format!(
            "read_u8: offset {} out of range (len={})",
            offset,
            data.len()
        )
    })
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16, String> {
    if offset + 2 > data.len() {
        return Err(format!(
            "read_u16: offset {} out of range (len={})",
            offset,
            data.len()
        ));
    }
    Ok(u16::from_be_bytes([data[offset], data[offset + 1]]))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, String> {
    if offset + 4 > data.len() {
        return Err(format!(
            "read_u32: offset {} out of range (len={})",
            offset,
            data.len()
        ));
    }
    Ok(u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64, String> {
    if offset + 8 > data.len() {
        return Err(format!(
            "read_u64: offset {} out of range (len={})",
            offset,
            data.len()
        ));
    }
    Ok(u64::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]))
}

/// Read a signed 32-bit integer stored in sign-magnitude format (GRIB2 convention).
fn read_signed_u32(data: &[u8], offset: usize) -> Result<i32, String> {
    let raw = read_u32(data, offset)?;
    let sign = (raw >> 31) & 1;
    let magnitude = raw & 0x7FFF_FFFF;
    if sign == 1 {
        Ok(-(magnitude as i32))
    } else {
        Ok(magnitude as i32)
    }
}

/// Read a signed 16-bit integer stored in sign-magnitude format.
fn read_signed_u16(data: &[u8], offset: usize) -> Result<i16, String> {
    let raw = read_u16(data, offset)?;
    let sign = (raw >> 15) & 1;
    let magnitude = raw & 0x7FFF;
    if sign == 1 {
        Ok(-(magnitude as i16))
    } else {
        Ok(magnitude as i16)
    }
}

fn read_f32(data: &[u8], offset: usize) -> Result<f32, String> {
    if offset + 4 > data.len() {
        return Err(format!(
            "read_f32: offset {} out of range (len={})",
            offset,
            data.len()
        ));
    }
    Ok(f32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

// ---------- section parsing ----------

impl Grib2File {
    /// Open a GRIB2 file from disk and parse it.
    pub fn open(path: &str) -> crate::Result<Self> {
        let data = std::fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Alias for `open` for compatibility.
    pub fn from_path(path: &str) -> crate::Result<Self> {
        Self::open(path)
    }

    /// Parse all GRIB2 messages from raw bytes.
    /// A GRIB2 file may contain multiple concatenated messages.
    /// Multi-field GRIB2 messages (where sections 3-7 repeat) are flattened
    /// into separate entries in the messages vec.
    pub fn from_bytes(data: &[u8]) -> crate::Result<Self> {
        let mut messages = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            if data.len() - pos < 20 {
                return Err(crate::GribError::Parse(format!(
                    "truncated GRIB2 envelope at byte {pos}"
                )));
            }
            if &data[pos..pos + 4] != b"GRIB" {
                return Err(crate::GribError::Parse(format!(
                    "non-GRIB bytes at envelope boundary {pos}"
                )));
            }
            let total_u64 = read_u64(data, pos + 8).map_err(crate::GribError::Parse)?;
            let total_len = usize::try_from(total_u64).map_err(|_| {
                crate::GribError::Parse(format!(
                    "GRIB2 envelope length {total_u64} is not addressable"
                ))
            })?;
            if total_len < 20 {
                return Err(crate::GribError::Parse(format!(
                    "GRIB2 envelope length {total_len} is below the 20-byte minimum"
                )));
            }
            let next = pos.checked_add(total_len).ok_or_else(|| {
                crate::GribError::Parse("GRIB2 envelope length overflows file offset".into())
            })?;
            if next > data.len() {
                return Err(crate::GribError::Parse(format!(
                    "GRIB2 envelope ending at {next} exceeds file length {}",
                    data.len()
                )));
            }
            if &data[next - 4..next] != b"7777" {
                return Err(crate::GribError::Parse(format!(
                    "GRIB2 envelope at {pos} lacks 7777 at its declared end"
                )));
            }
            let mut msgs = parse_message(data, pos).map_err(crate::GribError::Parse)?;
            messages.append(&mut msgs);
            pos = next;
        }

        Ok(Grib2File { messages })
    }

    /// Find the first message matching the given parameter and level.
    pub fn find(
        &self,
        category: u8,
        parameter: u8,
        level_type: u8,
        level: f64,
    ) -> Option<&Grib2Message> {
        self.messages.iter().find(|m| {
            m.product.parameter_category == category
                && m.product.parameter_number == parameter
                && m.product.level_type == level_type
                && (m.product.level_value - level).abs() < 0.5
        })
    }
}

/// Parse a single GRIB2 message envelope starting at `base`.
///
/// GRIB2 allows sections 3-7 to repeat within one message envelope, producing
/// multiple fields. Sections 0 and 1 are shared across all fields. When a new
/// section 4 is encountered after a complete field (one that already has a
/// section 4), the previous field is emitted and a new one begins. Section 3
/// (grid) is reused if not explicitly redefined before the next section 4.
fn parse_message(data: &[u8], base: usize) -> Result<Vec<Grib2Message>, String> {
    // --- Section 0 (Indicator) ---
    let discipline = read_u8(data, base + 6)?;
    let edition = read_u8(data, base + 7)?;
    if edition != 2 {
        return Err(format!("Unsupported GRIB edition: {}", edition));
    }
    let total_u64 = read_u64(data, base + 8)?;
    let total_length = usize::try_from(total_u64)
        .map_err(|_| format!("GRIB2 envelope length {total_u64} is not addressable"))?;
    if total_length < 20 {
        return Err(format!(
            "GRIB2 envelope length {total_length} is below the 20-byte minimum"
        ));
    }
    let msg_end = base
        .checked_add(total_length)
        .ok_or("GRIB2 envelope length overflows file offset")?;

    if msg_end > data.len() {
        return Err(format!(
            "Message extends beyond file: msg_end={}, file_len={}",
            msg_end,
            data.len()
        ));
    }
    if &data[msg_end - 4..msg_end] != b"7777" {
        return Err("GRIB2 envelope lacks 7777 at its declared end".into());
    }

    // Section 0 is 16 bytes
    let mut offset = base + 16;

    let mut reference_time = chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let mut identification = Identification::default();

    // Accumulator for the current field being built
    let mut grid = GridDefinition::default();
    let mut product = ProductDefinition::default();
    let mut data_rep = DataRepresentation::default();
    let mut bitmap: Option<Vec<bool>> = None;
    let mut raw_data: Vec<u8> = Vec::new();
    let mut seen_section1 = false;
    let mut grid_ready = false;
    // 0=between fields, 1=after Section 4, 2=after 5, 3=after 6.
    let mut field_stage = 0u8;

    // State carried across repeated field groups (Fix 2: bitmap reuse)
    let mut last_bitmap: Option<(Vec<bool>, GridDefinition)> = None;

    let mut messages: Vec<Grib2Message> = Vec::new();

    // Parse sections 1-8
    while offset < msg_end {
        // Check for "7777" end marker (Section 8)
        if offset + 4 <= msg_end && &data[offset..offset + 4] == b"7777" {
            break;
        }

        if offset + 5 > msg_end {
            return Err("Truncated section header".into());
        }

        let section_length = read_u32(data, offset)? as usize;
        let section_number = read_u8(data, offset + 4)?;

        if section_length < 5 || offset + section_length > msg_end {
            return Err(format!(
                "Invalid section {} length: {} at offset {}",
                section_number, section_length, offset
            ));
        }

        let sec = &data[offset..offset + section_length];

        match section_number {
            1 => {
                if seen_section1 || grid_ready || field_stage != 0 || !messages.is_empty() {
                    return Err("Section 1 is duplicated or out of order".into());
                }
                (identification, reference_time) = parse_section1(sec)?;
                seen_section1 = true;
            }
            2 => {
                if !seen_section1 || field_stage != 0 {
                    return Err("Section 2 is out of order".into());
                }
            }
            3 => {
                if !seen_section1 || field_stage != 0 {
                    return Err("Section 3 is missing Section 1 or interrupts a field".into());
                }
                grid = parse_section3(sec)?;
                grid_ready = true;
            }
            4 => {
                if !seen_section1 || !grid_ready || field_stage != 0 {
                    return Err("Section 4 is missing a grid or interrupts a field".into());
                }
                product = parse_section4(sec)?;
                field_stage = 1;
            }
            5 => {
                if field_stage != 1 {
                    return Err("Section 5 must immediately follow Section 4".into());
                }
                data_rep = parse_section5(sec)?;
                field_stage = 2;
            }
            6 => {
                if field_stage != 2 {
                    return Err("Section 6 must immediately follow Section 5".into());
                }
                bitmap = parse_section6(sec, &mut last_bitmap, &grid)?;
                field_stage = 3;
            }
            7 => {
                if field_stage != 3 {
                    return Err("Section 7 must immediately follow Section 6".into());
                }
                raw_data = parse_section7(sec);
                let grid_points = usize::try_from(grid.num_data_points)
                    .map_err(|_| "Section 3 point count is not addressable")?;
                if grid_points == 0 {
                    return Err("Section 3 declares zero data points".into());
                }
                let declared_values = data_rep.section5_num_data_points as usize;
                if let Some(ref mask) = bitmap {
                    let expected_bitmap_bits = ((grid_points + 7) / 8) * 8;
                    if mask.len() != expected_bitmap_bits
                        || mask[grid_points..].iter().any(|value| *value)
                    {
                        return Err(format!(
                            "Section 6 bitmap has invalid length/padding ({} bits); Section 3 declares {grid_points}",
                            mask.len()
                        ));
                    }
                    let normalized_mask = mask[..grid_points].to_vec();
                    let present = normalized_mask.iter().filter(|value| **value).count();
                    if present != declared_values {
                        return Err(format!(
                            "Section 6 bitmap has {present} present cells; Section 5 declares {declared_values} values"
                        ));
                    }
                    bitmap = Some(normalized_mask);
                } else if declared_values != grid_points {
                    return Err(format!(
                        "Section 5 declares {declared_values} values without a bitmap; Section 3 declares {grid_points}"
                    ));
                }
                messages.push(Grib2Message {
                    discipline,
                    identification: identification.clone(),
                    reference_time,
                    grid: grid.clone(),
                    product: std::mem::take(&mut product),
                    data_rep: std::mem::take(&mut data_rep),
                    bitmap: bitmap.take(),
                    raw_data: std::mem::take(&mut raw_data),
                });
                field_stage = 0;
            }
            _ => return Err(format!("unsupported or misplaced Section {section_number}")),
        }

        offset += section_length;
    }

    if offset != msg_end - 4 {
        return Err("Section 8 marker is not at the parsed envelope boundary".into());
    }
    if !seen_section1 || messages.is_empty() || field_stage != 0 {
        return Err("GRIB2 envelope is missing required Sections 1/3/4/5/6/7".into());
    }

    Ok(messages)
}

/// Parse Section 1 (Identification).
fn parse_section1(sec: &[u8]) -> Result<(Identification, NaiveDateTime), String> {
    if sec.len() < 21 {
        return Err("Section 1 too short".into());
    }
    let identification = Identification {
        center_id: read_u16(sec, 5)?,
        subcenter_id: read_u16(sec, 7)?,
        master_table_version: read_u8(sec, 9)?,
        local_table_version: read_u8(sec, 10)?,
        reference_time_significance: read_u8(sec, 11)?,
        production_status: read_u8(sec, 19)?,
        processed_data_type: read_u8(sec, 20)?,
    };
    let year = read_u16(sec, 12)? as i32;
    let month = read_u8(sec, 14)? as u32;
    let day = read_u8(sec, 15)? as u32;
    let hour = read_u8(sec, 16)? as u32;
    let minute = read_u8(sec, 17)? as u32;
    let second = read_u8(sec, 18)? as u32;

    let date = chrono::NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| format!("Invalid date: {}-{}-{}", year, month, day))?;
    let dt = date
        .and_hms_opt(hour, minute, second)
        .ok_or_else(|| format!("Invalid time: {}:{}:{}", hour, minute, second))?;
    Ok((identification, dt))
}

/// Parse Section 3 (Grid Definition).
fn parse_section3(sec: &[u8]) -> Result<GridDefinition, String> {
    if sec.len() < 14 {
        return Err("Section 3 too short".into());
    }
    let template = read_u16(sec, 12)?;

    let mut grid = GridDefinition::default();
    grid.template = template;

    // Number of data points is always at octets 7-10 (0-based: 6-9) for all templates.
    if sec.len() >= 10 {
        grid.num_data_points = read_u32(sec, 6)?;
    }
    // Shape of the Earth: octet 15 (0-based: 14)
    if sec.len() > 14 {
        grid.shape_of_earth = sec[14];
    }

    match template {
        0 => parse_grid_template_0(sec, &mut grid)?,
        1 => parse_grid_template_1(sec, &mut grid)?,
        10 => parse_grid_template_10(sec, &mut grid)?,
        20 => parse_grid_template_20(sec, &mut grid)?,
        30 => parse_grid_template_30(sec, &mut grid)?,
        40 => parse_grid_template_40(sec, &mut grid)?,
        90 => parse_grid_template_90(sec, &mut grid)?,
        _ => {
            // For unknown templates, try to extract basic dimensions if possible
            if sec.len() >= 38 {
                grid.nx = read_u32(sec, 30)?;
                grid.ny = read_u32(sec, 34)?;
            }
        }
    }

    Ok(grid)
}

/// Template 3.0: Latitude/Longitude (Equidistant Cylindrical).
fn parse_grid_template_0(sec: &[u8], grid: &mut GridDefinition) -> Result<(), String> {
    if sec.len() < 72 {
        return Err("Section 3 template 0 too short".into());
    }

    grid.nx = read_u32(sec, 30)?;
    grid.ny = read_u32(sec, 34)?;

    let basic_angle = read_u32(sec, 38)?;
    let subdivisions = read_u32(sec, 42)?;
    let divisor = if basic_angle == 0 || subdivisions == 0 {
        1_000_000.0
    } else {
        subdivisions as f64 / basic_angle as f64
    };

    grid.lat1 = read_signed_u32(sec, 46)? as f64 / divisor;
    grid.lon1 = read_signed_u32(sec, 50)? as f64 / divisor;
    grid.resolution_flags = read_u8(sec, 54)?;
    grid.lat2 = read_signed_u32(sec, 55)? as f64 / divisor;
    grid.lon2 = read_signed_u32(sec, 59)? as f64 / divisor;
    grid.dx = read_u32(sec, 63)? as f64 / divisor;
    grid.dy = read_u32(sec, 67)? as f64 / divisor;
    grid.scan_mode = read_u8(sec, 71)?;

    Ok(())
}

/// Template 3.30: Lambert Conformal.
fn parse_grid_template_30(sec: &[u8], grid: &mut GridDefinition) -> Result<(), String> {
    if sec.len() < 81 {
        return Err("Section 3 template 30 too short".into());
    }

    grid.nx = read_u32(sec, 30)?;
    grid.ny = read_u32(sec, 34)?;
    grid.lat1 = read_signed_u32(sec, 38)? as f64 / 1_000_000.0;
    grid.lon1 = read_signed_u32(sec, 42)? as f64 / 1_000_000.0;
    grid.resolution_flags = read_u8(sec, 46)?;
    grid.lov = read_signed_u32(sec, 51)? as f64 / 1_000_000.0;
    // Dx and Dy are stored in millimetres in GRIB2 template 3.30
    grid.dx = read_u32(sec, 55)? as f64 / 1000.0;
    grid.dy = read_u32(sec, 59)? as f64 / 1000.0;
    grid.scan_mode = read_u8(sec, 64)?;
    grid.latin1 = read_signed_u32(sec, 65)? as f64 / 1_000_000.0;
    grid.latin2 = read_signed_u32(sec, 69)? as f64 / 1_000_000.0;
    // South pole of projection (octets 74-80, 0-based: 73-79)
    if sec.len() >= 81 {
        grid.south_pole_lat = read_signed_u32(sec, 73)? as f64 / 1_000_000.0;
        grid.south_pole_lon = read_signed_u32(sec, 77)? as f64 / 1_000_000.0;
    }

    Ok(())
}

/// Template 3.10: Mercator.
/// WMO GRIB2 Section 3 Template 3.10 octet layout (1-based octets):
///   15: shape of earth, 30-33: Ni, 34-37: Nj,
///   38-41: La1, 42-45: Lo1, 46: resolution flags,
///   47-50: LaD, 51-54: La2, 55-58: Lo2,
///   59: scanning mode, 60: grid orientation angle,
///   61-64: Di (mm), 65-68: Dj (mm)
/// Offsets below are 0-based within the section bytes.
fn parse_grid_template_10(sec: &[u8], grid: &mut GridDefinition) -> Result<(), String> {
    if sec.len() < 72 {
        return Err("Section 3 template 10 (Mercator) too short".into());
    }

    grid.nx = read_u32(sec, 30)?;
    grid.ny = read_u32(sec, 34)?;
    grid.lat1 = read_signed_u32(sec, 38)? as f64 / 1_000_000.0;
    grid.lon1 = read_signed_u32(sec, 42)? as f64 / 1_000_000.0;
    // LaD - latitude where Dx/Dy are specified
    grid.lad = read_signed_u32(sec, 47)? as f64 / 1_000_000.0;
    grid.lat2 = read_signed_u32(sec, 51)? as f64 / 1_000_000.0;
    grid.lon2 = read_signed_u32(sec, 55)? as f64 / 1_000_000.0;
    grid.scan_mode = read_u8(sec, 59)?;
    // Di, Dj in millimetres
    grid.dx = read_u32(sec, 64)? as f64 / 1000.0;
    grid.dy = read_u32(sec, 68)? as f64 / 1000.0;

    Ok(())
}

/// Template 3.20: Polar Stereographic.
/// WMO GRIB2 Section 3 Template 3.20 octet layout (1-based octets):
///   15: shape of earth, 30-33: Nx, 34-37: Ny,
///   38-41: La1, 42-45: Lo1, 46: resolution/component flags,
///   47-50: LaD, 51-54: LoV, 55-58: Dx (mm), 59-62: Dy (mm),
///   63: projection centre flag, 64: scanning mode
fn parse_grid_template_20(sec: &[u8], grid: &mut GridDefinition) -> Result<(), String> {
    if sec.len() < 65 {
        return Err("Section 3 template 20 (Polar Stereographic) too short".into());
    }

    grid.nx = read_u32(sec, 30)?;
    grid.ny = read_u32(sec, 34)?;
    grid.lat1 = read_signed_u32(sec, 38)? as f64 / 1_000_000.0;
    grid.lon1 = read_signed_u32(sec, 42)? as f64 / 1_000_000.0;
    // LaD - true latitude (latitude where Dx/Dy are specified)
    grid.lad = read_signed_u32(sec, 47)? as f64 / 1_000_000.0;
    // LoV - orientation longitude (grid vertical longitude)
    grid.lov = read_signed_u32(sec, 51)? as f64 / 1_000_000.0;
    // Dx, Dy in millimetres
    grid.dx = read_u32(sec, 55)? as f64 / 1000.0;
    grid.dy = read_u32(sec, 59)? as f64 / 1000.0;
    grid.projection_center_flag = read_u8(sec, 63)?;
    grid.scan_mode = read_u8(sec, 64)?;

    Ok(())
}

/// Template 3.40: Gaussian Latitude/Longitude.
/// Same octet layout as Template 3.0 for most fields, but octet 73-76
/// contain N (number of parallels between pole and equator) instead of
/// the scanning mode appendix.
///
/// For reduced Gaussian grids, nx is set to 0xFFFFFFFF (or >= 0xFFFFFFFE)
/// and a pl array at the end of the section specifies points per latitude row.
fn parse_grid_template_40(sec: &[u8], grid: &mut GridDefinition) -> Result<(), String> {
    if sec.len() < 72 {
        return Err("Section 3 template 40 (Gaussian) too short".into());
    }

    grid.nx = read_u32(sec, 30)?;
    grid.ny = read_u32(sec, 34)?;

    let basic_angle = read_u32(sec, 38)?;
    let subdivisions = read_u32(sec, 42)?;
    let divisor = if basic_angle == 0 || subdivisions == 0 {
        1_000_000.0
    } else {
        subdivisions as f64 / basic_angle as f64
    };

    grid.lat1 = read_signed_u32(sec, 46)? as f64 / divisor;
    grid.lon1 = read_signed_u32(sec, 50)? as f64 / divisor;
    grid.lat2 = read_signed_u32(sec, 55)? as f64 / divisor;
    grid.lon2 = read_signed_u32(sec, 59)? as f64 / divisor;
    grid.dx = read_u32(sec, 63)? as f64 / divisor;
    // For Gaussian grids, octet 68-71 is N (number of parallels between
    // a pole and the equator), not Dy in the conventional sense.
    grid.n_parallel = read_u32(sec, 67)?;
    grid.scan_mode = read_u8(sec, 71)?;
    // Compute approximate dy for consumers that expect it
    grid.dy = if grid.n_parallel > 0 {
        90.0 / grid.n_parallel as f64
    } else if grid.ny > 1 {
        (grid.lat2 - grid.lat1).abs() / (grid.ny as f64 - 1.0)
    } else {
        0.0
    };

    // Check for reduced Gaussian grid: nx >= 0xFFFFFFFE indicates quasi-regular
    if grid.nx >= 0xFFFFFFFE {
        grid.is_reduced = true;
        // The pl array (number of points per latitude row) follows the standard
        // template bytes. Each entry is 2 bytes (u16), with ny entries.
        let pl_start = 72; // 0-based offset after scan_mode byte
        let nj = grid.ny as usize;
        if sec.len() >= pl_start + nj * 2 {
            let mut pl = Vec::with_capacity(nj);
            for row in 0..nj {
                let val = read_u16(sec, pl_start + row * 2)? as u32;
                pl.push(val);
            }
            grid.pl = Some(pl);
        }
    }

    Ok(())
}

/// Template 3.1: Rotated Latitude/Longitude.
/// Same basic layout as template 3.0 but with additional rotation parameters
/// at the end of the section.
fn parse_grid_template_1(sec: &[u8], grid: &mut GridDefinition) -> Result<(), String> {
    // First parse the regular lat/lon fields (same as template 0)
    parse_grid_template_0(sec, grid)?;

    // Rotated grid parameters start after the regular lat/lon fields.
    // Template 3.1 has the rotation parameters at octets 73-84 (0-based: 72-83).
    if sec.len() < 84 {
        return Err("Section 3 template 1 (Rotated Lat/Lon) too short".into());
    }

    grid.south_pole_lat = read_signed_u32(sec, 72)? as f64 / 1_000_000.0;
    grid.south_pole_lon = read_signed_u32(sec, 76)? as f64 / 1_000_000.0;
    grid.rotation_angle = read_f32(sec, 80)? as f64;

    Ok(())
}

/// Template 3.90: Space View Perspective or Orthographic.
/// Used for satellite imagery (e.g., GOES, Meteosat).
fn parse_grid_template_90(sec: &[u8], grid: &mut GridDefinition) -> Result<(), String> {
    if sec.len() < 72 {
        return Err("Section 3 template 90 (Space View) too short".into());
    }

    grid.nx = read_u32(sec, 30)?;
    grid.ny = read_u32(sec, 34)?;

    // Lap - latitude of sub-satellite point
    grid.satellite_lat = read_signed_u32(sec, 38)? as f64 / 1_000_000.0;
    // Lop - longitude of sub-satellite point
    grid.satellite_lon = read_signed_u32(sec, 42)? as f64 / 1_000_000.0;

    // Resolution and component flags at octet 47
    // Dx, Dy - apparent diameters in grid lengths
    grid.dx = read_u32(sec, 47)? as f64;
    grid.dy = read_u32(sec, 51)? as f64;

    // Xp, Yp - grid coordinates of sub-satellite point (scaled by 1000)
    grid.xp = read_u32(sec, 55)? as f64 / 1000.0;
    grid.yp = read_u32(sec, 59)? as f64 / 1000.0;

    grid.scan_mode = read_u8(sec, 63)?;

    // Altitude of the camera from the Earth's centre (in units of Earth's radius)
    // Nr - altitude scaled by 10^6
    if sec.len() >= 68 {
        let nr = read_u32(sec, 64)? as f64 / 1_000_000.0;
        // Convert from Earth-radii (from centre) to metres above surface
        let r_earth = 6_371_229.0;
        grid.altitude = (nr - 1.0) * r_earth;
    }

    Ok(())
}

/// Parse Section 4 (Product Definition).
fn parse_section4(sec: &[u8]) -> Result<ProductDefinition, String> {
    if sec.len() < 11 {
        return Err("Section 4 too short".into());
    }
    let template = read_u16(sec, 7)?;
    // PDT 4.0 contains two complete fixed-surface descriptors through octet
    // 34.  A shorter section must not inherit ProductDefinition defaults and
    // masquerade as an explicitly absent second surface.
    if template == 0 && sec.len() < 34 {
        return Err("Section 4 template 0 is truncated before the second fixed surface".into());
    }
    let mut prod = ProductDefinition::default();
    prod.template = template;

    // Templates 4.0, 4.1, 4.2, 4.8, etc. all share the first few bytes
    if sec.len() >= 28 {
        prod.parameter_category = read_u8(sec, 9)?;
        prod.parameter_number = read_u8(sec, 10)?;
        prod.generating_process = read_u8(sec, 11)?;
        prod.background_generating_process_id = read_u8(sec, 12)?;
        prod.forecast_generating_process_id = read_u8(sec, 13)?;
        prod.time_range_unit = read_u8(sec, 17)?;
        prod.forecast_time = read_u32(sec, 18)?;

        prod.level_type = read_u8(sec, 22)?;
        let scale_factor = read_u8(sec, 23)?;
        let scaled_value = read_u32(sec, 24)? as f64;
        if scale_factor < 128 {
            prod.level_value = scaled_value / 10.0_f64.powi(scale_factor as i32);
        } else {
            // sign-magnitude: MSB set means negative scale factor
            let neg_scale = (scale_factor & 0x7f) as i32;
            prod.level_value = scaled_value * 10.0_f64.powi(neg_scale);
        }

        if sec.len() >= 34 {
            prod.second_level_type = read_u8(sec, 28)?;
            let second_scale_factor = read_u8(sec, 29)?;
            let second_scaled_value = read_u32(sec, 30)? as f64;
            if prod.second_level_type != 255 {
                if second_scale_factor < 128 {
                    prod.second_level_value =
                        second_scaled_value / 10.0_f64.powi(second_scale_factor as i32);
                } else {
                    let neg_scale = (second_scale_factor & 0x7f) as i32;
                    prod.second_level_value = second_scaled_value * 10.0_f64.powi(neg_scale);
                }
            }
        }
    }

    // Template-specific parsing
    match template {
        1 => {
            // PDT 4.1: Individual ensemble forecast
            if sec.len() >= 37 {
                prod.ensemble_type = Some(read_u8(sec, 34)?);
                prod.perturbation_number = Some(read_u8(sec, 35)?);
                prod.num_forecasts_in_ensemble = Some(read_u8(sec, 36)?);
            }
        }
        2 => {
            // PDT 4.2: Derived forecasts based on all ensemble members
            if sec.len() >= 36 {
                prod.derived_forecast_type = Some(read_u8(sec, 34)?);
                prod.num_forecasts_in_ensemble = Some(read_u8(sec, 35)?);
            }
        }
        5 => {
            // PDT 4.5: Probability forecast at a point in time
            parse_pdt_probability_fields(sec, &mut prod, 34)?;
        }
        6 => {
            // PDT 4.6: Percentile forecast at a point in time
            parse_pdt_percentile_fields(sec, &mut prod, 34)?;
        }
        8 => {
            // PDT 4.8: Statistically processed values over a time interval
            parse_pdt_statistical_fields(sec, &mut prod, 34)?;
        }
        9 => {
            // PDT 4.9: Probability forecast over a time interval
            parse_pdt_probability_fields(sec, &mut prod, 34)?;
            parse_pdt_statistical_fields(sec, &mut prod, 47)?;
        }
        10 => {
            // PDT 4.10: Percentile forecast over a time interval
            parse_pdt_percentile_fields(sec, &mut prod, 34)?;
            parse_pdt_statistical_fields(sec, &mut prod, 35)?;
        }
        11 => {
            // PDT 4.11: Individual ensemble forecast + time interval
            // Ensemble fields first (same as 4.1)
            if sec.len() >= 37 {
                prod.ensemble_type = Some(read_u8(sec, 34)?);
                prod.perturbation_number = Some(read_u8(sec, 35)?);
                prod.num_forecasts_in_ensemble = Some(read_u8(sec, 36)?);
            }
            // Statistical fields follow at offset 37
            parse_pdt_statistical_fields(sec, &mut prod, 37)?;
        }
        12 => {
            // PDT 4.12: Derived ensemble forecast + time interval
            // Derived forecast fields first (same as 4.2)
            if sec.len() >= 36 {
                prod.derived_forecast_type = Some(read_u8(sec, 34)?);
                prod.num_forecasts_in_ensemble = Some(read_u8(sec, 35)?);
            }
            // Statistical fields follow at offset 36
            parse_pdt_statistical_fields(sec, &mut prod, 36)?;
        }
        _ => {}
    }

    Ok(prod)
}

/// Parse the probability metadata common to PDT 4.5 and 4.9.
/// `base` is the 0-based offset where the forecast probability number starts.
fn parse_pdt_probability_fields(
    sec: &[u8],
    prod: &mut ProductDefinition,
    base: usize,
) -> Result<(), String> {
    if sec.len() < base + 13 {
        return Ok(());
    }
    prod.probability_number = Some(read_u8(sec, base)?);
    prod.total_number_of_probabilities = Some(read_u8(sec, base + 1)?);
    prod.probability_type = Some(read_u8(sec, base + 2)?);
    prod.probability_lower_limit = read_scaled_optional(sec, base + 3, base + 4)?;
    prod.probability_upper_limit = read_scaled_optional(sec, base + 8, base + 9)?;
    Ok(())
}

/// Parse the percentile metadata common to PDT 4.6 and 4.10.
/// `offset` is the 0-based offset where the percentile value is stored.
fn parse_pdt_percentile_fields(
    sec: &[u8],
    prod: &mut ProductDefinition,
    offset: usize,
) -> Result<(), String> {
    if sec.len() > offset {
        prod.percentile_value = Some(read_u8(sec, offset)?);
    }
    Ok(())
}

fn read_scaled_optional(
    sec: &[u8],
    scale_offset: usize,
    value_offset: usize,
) -> Result<Option<f64>, String> {
    let scale_factor = read_u8(sec, scale_offset)?;
    let scaled_value = read_u32(sec, value_offset)?;
    if scale_factor == 255 || scaled_value == u32::MAX {
        return Ok(None);
    }
    let value = if scale_factor < 128 {
        scaled_value as f64 / 10.0_f64.powi(scale_factor as i32)
    } else {
        let neg_scale = 256 - scale_factor as i32;
        scaled_value as f64 * 10.0_f64.powi(neg_scale)
    };
    Ok(Some(value))
}

/// Parse the statistical time interval fields common to PDT 4.8, 4.11, 4.12.
/// `base` is the 0-based offset where the end-of-interval year starts.
fn parse_pdt_statistical_fields(
    sec: &[u8],
    prod: &mut ProductDefinition,
    base: usize,
) -> Result<(), String> {
    // End of overall time interval: year(2), month, day, hour, minute, second
    if sec.len() < base + 7 {
        return Ok(()); // Not enough data, skip gracefully
    }
    let year = read_u16(sec, base)? as i32;
    let month = read_u8(sec, base + 2)? as u32;
    let day = read_u8(sec, base + 3)? as u32;
    let hour = read_u8(sec, base + 4)? as u32;
    let minute = read_u8(sec, base + 5)? as u32;
    let second = read_u8(sec, base + 6)? as u32;

    if let Some(date) = chrono::NaiveDate::from_ymd_opt(year, month, day) {
        if let Some(dt) = date.and_hms_opt(hour, minute, second) {
            prod.end_of_interval = Some(dt);
        }
    }

    // base + 7: number of time range specifications (n)
    // base + 8..+11: number of missing values (u32)
    // Each time range spec is 12 bytes starting at base + 12
    if sec.len() < base + 12 {
        return Ok(());
    }
    let _n_specs = read_u8(sec, base + 7)?;

    // Parse the first time range specification
    let spec_base = base + 12;
    if sec.len() >= spec_base + 12 {
        prod.statistical_process_type = Some(read_u8(sec, spec_base)?);
        prod.statistical_time_range_unit = Some(read_u8(sec, spec_base + 2)?);
        // spec_base + 3..+6: length of time range (4 bytes)
        prod.time_range_length = Some(read_u32(sec, spec_base + 3)?);
    }

    Ok(())
}

/// Parse Section 5 (Data Representation).
fn parse_section5(sec: &[u8]) -> Result<DataRepresentation, String> {
    if sec.len() < 12 {
        return Err("Section 5 too short".into());
    }
    let template = read_u16(sec, 9)?;
    let mut dr = DataRepresentation::default();
    dr.template = template;
    // Bytes 6-9: number of data points (u32) — present in all DRS templates
    if sec.len() >= 10 {
        dr.section5_num_data_points = read_u32(sec, 5)?;
    }

    match template {
        0 => parse_drtemplate_simple(sec, &mut dr)?,
        2 => parse_drtemplate_complex(sec, &mut dr)?,
        3 => parse_drtemplate_complex_spatial(sec, &mut dr)?,
        4 => parse_drtemplate_simple(sec, &mut dr)?, // IEEE float (uses bits_per_value)
        40 => parse_drtemplate_simple(sec, &mut dr)?,
        41 => parse_drtemplate_simple(sec, &mut dr)?,
        42 => parse_drtemplate_ccsds(sec, &mut dr)?,
        50 | 51 => parse_drtemplate_simple(sec, &mut dr)?, // Spectral
        61 => parse_drtemplate_simple(sec, &mut dr)?,      // Simple with log pre-processing
        200 => parse_drtemplate_simple(sec, &mut dr)?,     // RLE (NCEP local)
        _ => {
            if sec.len() >= 20 {
                parse_drtemplate_simple(sec, &mut dr)?;
            }
        }
    }

    Ok(dr)
}

/// Common simple packing fields (Template 5.0, also base for 5.40, 5.41).
fn parse_drtemplate_simple(sec: &[u8], dr: &mut DataRepresentation) -> Result<(), String> {
    if sec.len() < 21 {
        return Err("Section 5 simple packing too short".into());
    }
    dr.reference_value = read_f32(sec, 11)?;
    dr.binary_scale = read_signed_u16(sec, 15)?;
    dr.decimal_scale = read_signed_u16(sec, 17)?;
    dr.bits_per_value = read_u8(sec, 19)?;
    dr.original_field_type = read_u8(sec, 20)?;
    Ok(())
}

/// Template 5.2: Complex packing.
fn parse_drtemplate_complex(sec: &[u8], dr: &mut DataRepresentation) -> Result<(), String> {
    parse_drtemplate_simple(sec, dr)?;
    if sec.len() < 47 {
        return Err("Section 5 complex packing too short".into());
    }
    dr.group_splitting_method = read_u8(sec, 21)?;
    dr.num_groups = read_u32(sec, 31)?;
    dr.group_width_ref = read_u8(sec, 35)?;
    dr.group_width_bits = read_u8(sec, 36)?;
    dr.group_length_ref = read_u32(sec, 37)?;
    dr.group_length_inc = read_u8(sec, 41)?;
    dr.last_group_length = read_u32(sec, 42)?;
    dr.group_length_bits = read_u8(sec, 46)?;
    Ok(())
}

/// Template 5.3: Complex packing with spatial differencing.
fn parse_drtemplate_complex_spatial(sec: &[u8], dr: &mut DataRepresentation) -> Result<(), String> {
    parse_drtemplate_complex(sec, dr)?;
    if sec.len() < 49 {
        return Err("Section 5 complex+spatial too short".into());
    }
    dr.spatial_diff_order = read_u8(sec, 47)?;
    dr.spatial_diff_bytes = read_u8(sec, 48)?;
    Ok(())
}

/// Template 5.42: CCSDS (AEC/SZIP) packing.
fn parse_drtemplate_ccsds(sec: &[u8], dr: &mut DataRepresentation) -> Result<(), String> {
    parse_drtemplate_simple(sec, dr)?;
    if sec.len() < 25 {
        return Err("Section 5 CCSDS packing too short".into());
    }
    // Byte 20: type of original field values (not stored, skip)
    // Byte 21: CCSDS compression options mask (single byte, GRIB2 octet 22)
    dr.ccsds_flags = read_u8(sec, 21)? as u16;
    // Byte 22: block size (single byte, GRIB2 octet 23)
    dr.ccsds_block_size = read_u8(sec, 22)? as u16;
    // Bytes 23-24: reference sample interval (u16, GRIB2 octets 24-25)
    dr.ccsds_rsi = read_u16(sec, 23)?;
    Ok(())
}

/// Parse Section 6 (Bitmap).
///
/// `last_bitmap` is the most recently parsed bitmap, used when indicator == 254
/// ("use previously defined bitmap").
fn parse_section6(
    sec: &[u8],
    last_bitmap: &mut Option<(Vec<bool>, GridDefinition)>,
    grid: &GridDefinition,
) -> Result<Option<Vec<bool>>, String> {
    if sec.len() < 6 {
        return Err("Section 6 too short".into());
    }
    let indicator = read_u8(sec, 5)?;
    if indicator == 255 {
        // A no-bitmap field invalidates the previously saved mask.  A later
        // 254 must not resurrect an older field's bitmap.
        *last_bitmap = None;
        return Ok(None);
    }
    if indicator == 0 {
        let bitmap_bytes = &sec[6..];
        let mut bits = Vec::with_capacity(bitmap_bytes.len() * 8);
        for &byte in bitmap_bytes {
            for bit in (0..8).rev() {
                bits.push((byte >> bit) & 1 == 1);
            }
        }
        *last_bitmap = Some((bits.clone(), grid.clone()));
        return Ok(Some(bits));
    }
    if indicator == 254 {
        // Reuse previously defined bitmap
        let (bits, bitmap_grid) = last_bitmap
            .as_ref()
            .ok_or_else(|| "Section 6 requests bitmap 254 without a previous bitmap".to_string())?;
        if bitmap_grid != grid {
            return Err("Section 6 bitmap 254 crosses an incompatible Section 3 grid".into());
        }
        return Ok(Some(bits.clone()));
    }
    Err(format!(
        "unsupported Section 6 bitmap indicator {indicator}"
    ))
}

/// Parse Section 7 (Data) - extract raw data bytes.
fn parse_section7(sec: &[u8]) -> Vec<u8> {
    if sec.len() <= 5 {
        return Vec::new();
    }
    sec[5..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_section1_preserves_origin_and_production_identity() {
        let mut section = vec![0u8; 21];
        section[0..4].copy_from_slice(&21u32.to_be_bytes());
        section[4] = 1;
        section[5..7].copy_from_slice(&7u16.to_be_bytes());
        section[7..9].copy_from_slice(&0u16.to_be_bytes());
        section[9] = 2;
        section[10] = 1;
        section[11] = 1;
        section[12..14].copy_from_slice(&2026u16.to_be_bytes());
        section[14] = 7;
        section[15] = 20;
        section[16] = 0;
        section[19] = 0;
        section[20] = 1;

        let (identity, time) = parse_section1(&section).unwrap();
        assert_eq!(identity.center_id, 7);
        assert_eq!(identity.subcenter_id, 0);
        assert_eq!(identity.master_table_version, 2);
        assert_eq!(identity.local_table_version, 1);
        assert_eq!(identity.reference_time_significance, 1);
        assert_eq!(identity.production_status, 0);
        assert_eq!(identity.processed_data_type, 1);
        assert_eq!(time.to_string(), "2026-07-20 00:00:00");
    }

    #[test]
    fn pdt_zero_rejects_a_truncated_second_surface_descriptor() {
        let mut section = vec![0u8; 33];
        section[0..4].copy_from_slice(&33u32.to_be_bytes());
        section[4] = 4;
        section[7..9].copy_from_slice(&0u16.to_be_bytes());
        assert!(parse_section4(&section)
            .unwrap_err()
            .contains("truncated before the second fixed surface"));
    }

    #[test]
    fn drt_zero_requires_and_preserves_original_field_type() {
        let mut truncated = vec![0u8; 20];
        truncated[7..9].copy_from_slice(&0u16.to_be_bytes());
        assert!(parse_section5(&truncated)
            .unwrap_err()
            .contains("simple packing too short"));

        let mut complete = vec![0u8; 21];
        complete[7..9].copy_from_slice(&0u16.to_be_bytes());
        complete[20] = 1;
        assert_eq!(parse_section5(&complete).unwrap().original_field_type, 1);
    }

    #[test]
    fn bitmap_reuse_retains_raw_padding_for_a_non_byte_aligned_grid() {
        let explicit = [0, 0, 0, 8, 6, 0, 0xff, 0x80];
        let grid = GridDefinition {
            nx: 9,
            ny: 1,
            num_data_points: 9,
            ..GridDefinition::default()
        };
        let mut prior = None;
        let raw = parse_section6(&explicit, &mut prior, &grid)
            .unwrap()
            .unwrap();
        assert_eq!(raw.len(), 16);
        assert!(raw[..9].iter().all(|value| *value));
        assert!(raw[9..].iter().all(|value| !*value));

        let reused = [0, 0, 0, 6, 6, 254];
        assert_eq!(
            parse_section6(&reused, &mut prior, &grid).unwrap(),
            Some(raw)
        );
        let mut empty = None;
        assert!(parse_section6(&reused, &mut empty, &grid)
            .unwrap_err()
            .contains("without a previous bitmap"));

        let incompatible = GridDefinition {
            nx: 16,
            ny: 1,
            num_data_points: 16,
            ..GridDefinition::default()
        };
        let mut incompatible_prior = Some((vec![true; 16], grid.clone()));
        assert!(
            parse_section6(&reused, &mut incompatible_prior, &incompatible)
                .unwrap_err()
                .contains("incompatible Section 3 grid")
        );

        let explicit_byte = [0, 0, 0, 7, 6, 0, 0xff];
        let no_bitmap = [0, 0, 0, 6, 6, 255];
        let mut stale = None;
        parse_section6(&explicit_byte, &mut stale, &grid).unwrap();
        assert!(stale.is_some());
        assert_eq!(parse_section6(&no_bitmap, &mut stale, &grid).unwrap(), None);
        assert!(stale.is_none());
        assert!(parse_section6(&reused, &mut stale, &grid)
            .unwrap_err()
            .contains("without a previous bitmap"));
    }

    fn seed_common_section4(sec: &mut [u8], template: u16) {
        sec[7..9].copy_from_slice(&template.to_be_bytes());
        sec[9] = 0;
        sec[10] = 0;
        sec[17] = 1;
        sec[18..22].copy_from_slice(&24u32.to_be_bytes());
        sec[22] = 103;
        sec[23] = 0;
        sec[24..28].copy_from_slice(&2u32.to_be_bytes());
    }

    fn seed_statistical_window(sec: &mut [u8], base: usize, length_hours: u32) {
        sec[base..base + 2].copy_from_slice(&2026u16.to_be_bytes());
        sec[base + 2] = 5;
        sec[base + 3] = 7;
        sec[base + 4] = 0;
        sec[base + 5] = 0;
        sec[base + 6] = 0;
        sec[base + 7] = 1;
        let spec_base = base + 12;
        sec[spec_base] = 1;
        sec[spec_base + 2] = 1;
        sec[spec_base + 3..spec_base + 7].copy_from_slice(&length_hours.to_be_bytes());
    }

    #[test]
    fn parse_section4_uses_sign_magnitude_for_both_fixed_surface_scales() {
        let mut section = vec![0u8; 34];
        seed_common_section4(&mut section, 0);
        section[23] = 0x81;
        section[24..28].copy_from_slice(&1u32.to_be_bytes());
        section[28] = 106;
        section[29] = 0x82;
        section[30..34].copy_from_slice(&3u32.to_be_bytes());

        let product = parse_section4(&section).unwrap();
        assert_eq!(product.level_value, 10.0);
        assert_eq!(product.second_level_value, 300.0);
    }

    #[test]
    fn parse_section4_probability_template_captures_threshold() {
        let mut sec = vec![0u8; 47];
        seed_common_section4(&mut sec, 5);
        sec[34] = 0;
        sec[35] = 26;
        sec[36] = 2;
        sec[37] = 3;
        sec[38..42].copy_from_slice(&305_372u32.to_be_bytes());
        sec[42] = 255;
        sec[43..47].copy_from_slice(&u32::MAX.to_be_bytes());

        let product = parse_section4(&sec).expect("section 4 should parse");

        assert_eq!(product.template, 5);
        assert_eq!(product.probability_number, Some(0));
        assert_eq!(product.total_number_of_probabilities, Some(26));
        assert_eq!(product.probability_type, Some(2));
        assert_eq!(product.probability_lower_limit, Some(305.372));
        assert_eq!(product.probability_upper_limit, None);
    }

    #[test]
    fn parse_section4_percentile_template_captures_percentile() {
        let mut sec = vec![0u8; 35];
        seed_common_section4(&mut sec, 6);
        sec[34] = 50;

        let product = parse_section4(&sec).expect("section 4 should parse");

        assert_eq!(product.template, 6);
        assert_eq!(product.percentile_value, Some(50));
    }

    #[test]
    fn parse_section4_interval_probability_template_captures_threshold_and_window() {
        let mut sec = vec![0u8; 71];
        seed_common_section4(&mut sec, 9);
        sec[34] = 1;
        sec[35] = 26;
        sec[36] = 1;
        sec[37] = 255;
        sec[38..42].copy_from_slice(&u32::MAX.to_be_bytes());
        sec[42] = 0;
        sec[43..47].copy_from_slice(&300u32.to_be_bytes());
        seed_statistical_window(&mut sec, 47, 6);

        let product = parse_section4(&sec).expect("section 4 should parse");

        assert_eq!(product.template, 9);
        assert_eq!(product.probability_type, Some(1));
        assert_eq!(product.probability_lower_limit, None);
        assert_eq!(product.probability_upper_limit, Some(300.0));
        assert_eq!(product.statistical_time_range_hours(), Some(6));
    }

    #[test]
    fn parse_section4_interval_percentile_template_captures_percentile_and_window() {
        let mut sec = vec![0u8; 59];
        seed_common_section4(&mut sec, 10);
        sec[34] = 90;
        seed_statistical_window(&mut sec, 35, 3);

        let product = parse_section4(&sec).expect("section 4 should parse");

        assert_eq!(product.template, 10);
        assert_eq!(product.percentile_value, Some(90));
        assert_eq!(product.statistical_time_range_hours(), Some(3));
    }

    #[test]
    fn parse_section4_statistical_window_uses_spec_unit_and_length() {
        let mut sec = vec![0u8; 58];
        sec[7..9].copy_from_slice(&8u16.to_be_bytes());
        sec[9] = 1;
        sec[10] = 8;
        sec[17] = 0;
        sec[18..22].copy_from_slice(&6u32.to_be_bytes());
        sec[22] = 1;
        sec[24..28].copy_from_slice(&0u32.to_be_bytes());

        let base = 34;
        sec[base..base + 2].copy_from_slice(&2026u16.to_be_bytes());
        sec[base + 2] = 4;
        sec[base + 3] = 15;
        sec[base + 4] = 6;
        sec[base + 5] = 0;
        sec[base + 6] = 0;
        sec[base + 7] = 1;

        let spec_base = base + 12;
        sec[spec_base] = 1;
        sec[spec_base + 1] = 2;
        sec[spec_base + 2] = 1;
        sec[spec_base + 3..spec_base + 7].copy_from_slice(&6u32.to_be_bytes());
        sec[spec_base + 7] = 1;
        sec[spec_base + 8..spec_base + 12].copy_from_slice(&1u32.to_be_bytes());

        let product = parse_section4(&sec).expect("section 4 should parse");

        assert_eq!(product.template, 8);
        assert_eq!(product.time_range_unit, 0);
        assert_eq!(product.statistical_process_type, Some(1));
        assert_eq!(product.statistical_time_range_unit, Some(1));
        assert_eq!(product.time_range_length, Some(6));
        assert_eq!(product.statistical_time_range_hours(), Some(6));
    }
}
