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
    pub reference_time: NaiveDateTime,
    pub grid: GridDefinition,
    pub product: ProductDefinition,
    pub data_rep: DataRepresentation,
    pub bitmap: Option<Vec<bool>>,
    pub raw_data: Vec<u8>,
}

/// Grid definition from Section 3.
#[derive(Debug, Clone)]
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
    pub forecast_time: u32,
    pub time_range_unit: u8,
    pub level_type: u8,
    pub level_value: f64,
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
            forecast_time: 0,
            time_range_unit: 0,
            level_type: 0,
            level_value: 0.0,
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

        while pos + 16 <= data.len() {
            // Scan for "GRIB" magic
            match find_magic(data, pos) {
                Some(p) => pos = p,
                None => break,
            }

            let mut msgs = parse_message(data, pos).map_err(crate::GribError::Parse)?;
            let total_len = read_u64(data, pos + 8).map_err(crate::GribError::Parse)? as usize;
            messages.append(&mut msgs);

            // Advance past this message
            pos += total_len;
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

/// Scan forward from `start` to find the "GRIB" magic bytes.
fn find_magic(data: &[u8], start: usize) -> Option<usize> {
    let end = if data.len() >= 4 { data.len() - 3 } else { 0 };
    let mut i = start;
    while i < end {
        if &data[i..i + 4] == b"GRIB" {
            return Some(i);
        }
        i += 1;
    }
    None
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
    let total_length = read_u64(data, base + 8)? as usize;
    let msg_end = base + total_length;

    if msg_end > data.len() {
        return Err(format!(
            "Message extends beyond file: msg_end={}, file_len={}",
            msg_end,
            data.len()
        ));
    }

    // Section 0 is 16 bytes
    let mut offset = base + 16;

    let mut reference_time = chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();

    // Accumulator for the current field being built
    let mut grid = GridDefinition::default();
    let mut product = ProductDefinition::default();
    let mut data_rep = DataRepresentation::default();
    let mut bitmap: Option<Vec<bool>> = None;
    let mut raw_data: Vec<u8> = Vec::new();
    let mut has_section4 = false;

    // State carried across repeated field groups (Fix 2: bitmap reuse)
    let mut last_bitmap: Option<Vec<bool>> = None;

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
            1 => reference_time = parse_section1(sec)?,
            2 => { /* Local Use - skip */ }
            3 => {
                grid = parse_section3(sec)?;
            }
            4 => {
                // A new section 4 while we already have one means the previous
                // field set (3+4+5+6+7) is complete -- emit it.
                if has_section4 {
                    messages.push(Grib2Message {
                        discipline,
                        reference_time,
                        grid: grid.clone(),
                        product,
                        data_rep,
                        bitmap: bitmap.clone(),
                        raw_data,
                    });
                    // Reset per-field state; keep grid (reused if no new section 3)
                    data_rep = DataRepresentation::default();
                    bitmap = None;
                    raw_data = Vec::new();
                }
                product = parse_section4(sec)?;
                has_section4 = true;
            }
            5 => data_rep = parse_section5(sec)?,
            6 => {
                bitmap = parse_section6(sec, &last_bitmap)?;
                // Track the last bitmap for indicator 254 reuse
                if bitmap.is_some() {
                    last_bitmap = bitmap.clone();
                }
            }
            7 => raw_data = parse_section7(sec),
            _ => { /* Unknown section - skip */ }
        }

        offset += section_length;
    }

    // Emit the final (or only) field, but only if a valid section 4 was encountered.
    // Without a section 4, we have no product definition and would emit a phantom message.
    if has_section4 {
        messages.push(Grib2Message {
            discipline,
            reference_time,
            grid,
            product,
            data_rep,
            bitmap,
            raw_data,
        });
    }

    Ok(messages)
}

/// Parse Section 1 (Identification).
fn parse_section1(sec: &[u8]) -> Result<NaiveDateTime, String> {
    if sec.len() < 19 {
        return Err("Section 1 too short".into());
    }
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
    Ok(dt)
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
    let mut prod = ProductDefinition::default();
    prod.template = template;

    // Templates 4.0, 4.1, 4.2, 4.8, etc. all share the first few bytes
    if sec.len() >= 28 {
        prod.parameter_category = read_u8(sec, 9)?;
        prod.parameter_number = read_u8(sec, 10)?;
        prod.generating_process = read_u8(sec, 11)?;
        prod.time_range_unit = read_u8(sec, 17)?;
        prod.forecast_time = read_u32(sec, 18)?;

        prod.level_type = read_u8(sec, 22)?;
        let scale_factor = read_u8(sec, 23)?;
        let scaled_value = read_u32(sec, 24)? as f64;
        if scale_factor < 128 {
            prod.level_value = scaled_value / 10.0_f64.powi(scale_factor as i32);
        } else {
            // sign-magnitude: MSB set means negative scale factor
            let neg_scale = 256 - scale_factor as i32;
            prod.level_value = scaled_value * 10.0_f64.powi(neg_scale);
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
    if sec.len() < 20 {
        return Err("Section 5 simple packing too short".into());
    }
    dr.reference_value = read_f32(sec, 11)?;
    dr.binary_scale = read_signed_u16(sec, 15)?;
    dr.decimal_scale = read_signed_u16(sec, 17)?;
    dr.bits_per_value = read_u8(sec, 19)?;
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
    last_bitmap: &Option<Vec<bool>>,
) -> Result<Option<Vec<bool>>, String> {
    if sec.len() < 6 {
        return Err("Section 6 too short".into());
    }
    let indicator = read_u8(sec, 5)?;
    if indicator == 255 {
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
        return Ok(Some(bits));
    }
    if indicator == 254 {
        // Reuse previously defined bitmap
        return Ok(last_bitmap.clone());
    }
    // Other indicator values
    Ok(None)
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
