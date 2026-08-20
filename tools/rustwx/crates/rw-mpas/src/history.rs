//! Reading one MPAS history frame, plus the pinned static geometry it needs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::error::{MpasError, MpasResult};
use crate::fieldmap::{FieldMapping, FieldSource, FIELD_MAP};
use crate::weights::MeshCoordinates;

/// A calendar instant, to the second. The frame carries one and stamps it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

impl Timestamp {
    /// Parse the `YYYY-MM-DD_HH.MM.SS` (or `:`, or `T`) stamp MPAS puts in a
    /// history file name.
    pub fn parse(text: &str) -> MpasResult<Timestamp> {
        let bytes: Vec<char> = text.chars().collect();
        for start in 0..bytes.len().saturating_sub(18) {
            let slice: String = bytes[start..start + 19].iter().collect();
            if let Some(stamp) = Timestamp::parse_exact(&slice) {
                return Ok(stamp);
            }
        }
        Err(MpasError::Refusal(format!(
            "no YYYY-MM-DD_HH.MM.SS timestamp in '{text}'"
        )))
    }

    fn parse_exact(s: &str) -> Option<Timestamp> {
        let b: Vec<char> = s.chars().collect();
        if b.len() != 19 {
            return None;
        }
        let digits = |from: usize, n: usize| -> Option<u32> {
            let mut value = 0u32;
            for c in &b[from..from + n] {
                value = value * 10 + c.to_digit(10)?;
            }
            Some(value)
        };
        if b[4] != '-' || b[7] != '-' {
            return None;
        }
        if !matches!(b[10], '_' | 'T') {
            return None;
        }
        if !matches!(b[13], '.' | ':') || !matches!(b[16], '.' | ':') {
            return None;
        }
        Some(Timestamp {
            year: digits(0, 4)? as i32,
            month: digits(5, 2)?,
            day: digits(8, 2)?,
            hour: digits(11, 2)?,
            minute: digits(14, 2)?,
            second: digits(17, 2)?,
        })
    }

    /// The 19-byte WRF label, which is what `Times` and the date attributes
    /// carry.
    pub fn label(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}_{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    pub fn iso(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }

    /// Days since the proleptic-Gregorian epoch, for a difference in seconds.
    fn day_number(&self) -> i64 {
        let (y, m) = if self.month <= 2 {
            (self.year as i64 - 1, self.month as i64 + 12)
        } else {
            (self.year as i64, self.month as i64)
        };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = y - era * 400;
        let doy = (153 * (m - 3) + 2) / 5 + self.day as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    pub fn seconds_since(&self, other: &Timestamp) -> i64 {
        let days = self.day_number() - other.day_number();
        let secs = (self.hour as i64 * 3600 + self.minute as i64 * 60 + self.second as i64)
            - (other.hour as i64 * 3600 + other.minute as i64 * 60 + other.second as i64);
        days * 86_400 + secs
    }
}

/// One MPAS history frame plus the pinned static geometry it needs.
#[derive(Debug)]
pub struct MpasFrame {
    pub history_path: PathBuf,
    pub history_sha256: String,
    pub init_path: Option<PathBuf>,
    pub init_sha256: Option<String>,
    pub valid_time: Timestamp,
    pub n_cells: usize,
    pub n_levels: usize,
    pub n_soil_levels: usize,
    /// WRF name -> (values as `(n_cells, levels)` row-major, levels).
    pub fields: BTreeMap<String, (Vec<f32>, usize)>,
    pub latitude_degrees: Vec<f64>,
    pub longitude_degrees: Vec<f64>,
    pub absent: Vec<String>,
}

/// Read one cell-indexed field, dropping a leading single `Time` record.
fn read_cell_field(
    file: &netcrust::File,
    name: &str,
    n_cells: usize,
) -> MpasResult<Option<(Vec<f32>, usize)>> {
    let Some(variable) = file.variable(name) else {
        return Ok(None);
    };
    let has_time = variable
        .dimensions()
        .first()
        .map(|d| d.name() == "Time")
        .unwrap_or(false);
    let shape: Vec<usize> = variable.shape();
    if has_time && shape.first() != Some(&1) {
        return Err(MpasError::Refusal(format!(
            "{name} carries {} time records; expected exactly 1",
            shape.first().copied().unwrap_or(0)
        )));
    }
    let array = file.read_array_f64(name)?;
    let values = array.into_values();
    let trailing: usize = if has_time { shape[1..].iter().product() } else { shape.iter().product() };
    if values.len() < trailing {
        return Err(MpasError::Refusal(format!(
            "{name} read back {} value(s), fewer than the {trailing} its shape declares",
            values.len()
        )));
    }
    let slice = &values[values.len() - trailing..];
    if slice.len() % n_cells != 0 {
        return Err(MpasError::Refusal(format!(
            "{name} has {} value(s), not a whole number of levels over {n_cells} cells",
            slice.len()
        )));
    }
    let levels = slice.len() / n_cells;
    let mut out = Vec::with_capacity(slice.len());
    for &v in slice {
        let f = v as f32;
        if !f.is_finite() {
            return Err(MpasError::Refusal(format!("history {name} is not finite")));
        }
        out.push(f);
    }
    Ok(Some((out, levels)))
}

/// Read a history frame and, when given, the pinned init file beside it.
pub fn read_history(
    history_path: &Path,
    init_path: Option<&Path>,
    valid_time: Option<Timestamp>,
    history_sha256: Option<&str>,
    init_sha256: Option<&str>,
) -> MpasResult<MpasFrame> {
    let digest = crate::sha256_file(history_path)?;
    if let Some(want) = history_sha256 {
        if digest != want {
            return Err(MpasError::Refusal(format!(
                "history {} is sha256 {digest}, expected {want}",
                history_path.display()
            )));
        }
    }

    let mut fields: BTreeMap<String, (Vec<f32>, usize)> = BTreeMap::new();
    let mut absent: Vec<String> = Vec::new();

    let file = netcrust::File::open(history_path)?;
    let n_cells = file
        .dimension("nCells")
        .ok_or_else(|| MpasError::Refusal("history has no nCells dimension".to_string()))?
        .len();
    let n_levels = file
        .dimension("nVertLevels")
        .ok_or_else(|| MpasError::Refusal("history has no nVertLevels dimension".to_string()))?
        .len();
    let n_soil = file.dimension("nSoilLevels").map(|d| d.len()).unwrap_or(0);
    for name in ["latCell", "lonCell"] {
        if file.variable(name).is_none() {
            return Err(MpasError::Refusal(format!("history has no {name}")));
        }
    }
    let latitude = file.read_f64("latCell")?;
    let longitude = file.read_f64("lonCell")?;
    for mapping in FIELD_MAP.iter() {
        if mapping.source != FieldSource::History {
            continue;
        }
        let Some(mpas_name) = mapping.mpas_name else {
            continue;
        };
        match read_cell_field(&file, mpas_name, n_cells)? {
            Some(value) => {
                fields.insert(mapping.wrf_name.to_string(), value);
            }
            None => absent.push(mapping.wrf_name.to_string()),
        }
    }
    drop(file);

    let mut init_resolved: Option<PathBuf> = None;
    let mut init_digest: Option<String> = None;
    match init_path {
        Some(path) => {
            let d = crate::sha256_file(path)?;
            if let Some(want) = init_sha256 {
                if d != want {
                    return Err(MpasError::Refusal(format!(
                        "init {} is sha256 {d}, expected {want}",
                        path.display()
                    )));
                }
            }
            let init = netcrust::File::open(path)?;
            let init_cells = init
                .dimension("nCells")
                .ok_or_else(|| MpasError::Refusal("init has no nCells dimension".to_string()))?
                .len();
            if init_cells != n_cells {
                return Err(MpasError::Refusal(format!(
                    "init and history disagree on nCells ({init_cells} vs {n_cells})"
                )));
            }
            for mapping in FIELD_MAP.iter() {
                if mapping.source != FieldSource::Init {
                    continue;
                }
                let Some(mpas_name) = mapping.mpas_name else {
                    continue;
                };
                match read_cell_field(&init, mpas_name, n_cells)? {
                    Some(value) => {
                        fields.insert(mapping.wrf_name.to_string(), value);
                    }
                    None => absent.push(mapping.wrf_name.to_string()),
                }
            }
            init_resolved = Some(path.to_path_buf());
            init_digest = Some(d);
        }
        None => {
            for mapping in FIELD_MAP.iter() {
                if mapping.source == FieldSource::Init {
                    absent.push(mapping.wrf_name.to_string());
                }
            }
        }
    }

    if let Some((_, levels)) = fields.get("PHB") {
        if *levels != n_levels + 1 {
            return Err(MpasError::Refusal(format!(
                "init zgrid carries {levels} levels; expected {}",
                n_levels + 1
            )));
        }
    }

    let resolved_time = match valid_time {
        Some(t) => t,
        None => Timestamp::parse(
            history_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default(),
        )?,
    };

    let deg = 180.0 / std::f64::consts::PI;
    Ok(MpasFrame {
        history_path: history_path.to_path_buf(),
        history_sha256: digest,
        init_path: init_resolved,
        init_sha256: init_digest,
        valid_time: resolved_time,
        n_cells,
        n_levels,
        n_soil_levels: n_soil,
        fields,
        latitude_degrees: latitude.iter().map(|v| v * deg).collect(),
        longitude_degrees: longitude
            .iter()
            .map(|v| (v * deg + 180.0).rem_euclid(360.0) - 180.0)
            .collect(),
        absent,
    })
}

/// Refuse unless the history's own cell centres are the weights' mesh.
///
/// This is the guard against the one bug that would silently ruin every map:
/// resampling one mesh's fields through another mesh's index map, or through
/// a de-rotated copy of a `grid_rotate` output. The rotated grid stores
/// post-rotation earth coordinates, so a correct pairing agrees to float32
/// round-off; anything else is a different mesh and is refused here rather
/// than discovered by eye in a flipped map.
pub fn assert_mesh_agrees(
    frame: &MpasFrame,
    mesh: &MeshCoordinates,
    tolerance_degrees: f64,
) -> MpasResult<(f64, f64)> {
    if mesh.n_cells() != frame.n_cells {
        return Err(MpasError::Refusal(format!(
            "mesh has {} cells; history has {}",
            mesh.n_cells(),
            frame.n_cells
        )));
    }
    let mut latitude_error: f64 = 0.0;
    let mut longitude_error: f64 = 0.0;
    for i in 0..frame.n_cells {
        let dlat = (mesh.latitude_degrees[i] - frame.latitude_degrees[i]).abs();
        if dlat > latitude_error {
            latitude_error = dlat;
        }
        let raw = (mesh.longitude_degrees[i] - frame.longitude_degrees[i]).abs();
        let dlon = raw.min(360.0 - raw);
        if dlon > longitude_error {
            longitude_error = dlon;
        }
    }
    if latitude_error > tolerance_degrees || longitude_error > tolerance_degrees {
        return Err(MpasError::Refusal(format!(
            "history cell centres do not match the weights mesh (max |dlat|={latitude_error:e} deg, max |dlon|={longitude_error:e} deg)"
        )));
    }
    Ok((latitude_error, longitude_error))
}

/// The mapping's declared level count for this frame, or `None` when the
/// frame cannot carry it at all.
pub fn expected_levels(mapping: &FieldMapping, frame: &MpasFrame) -> Option<usize> {
    match mapping.rank {
        crate::fieldmap::Rank::Surface => Some(1),
        crate::fieldmap::Rank::Mass3d | crate::fieldmap::Rank::U3d | crate::fieldmap::Rank::V3d => {
            Some(frame.n_levels)
        }
        crate::fieldmap::Rank::W3d => Some(frame.n_levels + 1),
        crate::fieldmap::Rank::Soil => {
            if frame.n_soil_levels == 0 {
                None
            } else {
                Some(frame.n_soil_levels)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_history_file_name_yields_its_valid_time() {
        let t = Timestamp::parse("cuda-history.2026-08-10_13.00.00.nc").unwrap();
        assert_eq!(t.label(), "2026-08-10_13:00:00");
        assert_eq!(t.iso(), "2026-08-10T13:00:00");
    }

    #[test]
    fn a_name_without_a_stamp_is_refused() {
        let err = Timestamp::parse("history.nc").unwrap_err().to_string();
        assert!(err.contains("no YYYY-MM-DD"), "{err}");
    }

    #[test]
    fn a_lead_time_is_the_difference_in_seconds() {
        let origin = Timestamp::parse("2026-08-10_12:00:00").unwrap();
        let valid = Timestamp::parse("2026-08-10_13:00:00").unwrap();
        assert_eq!(valid.seconds_since(&origin), 3600);
        let next_day = Timestamp::parse("2026-08-11_00:00:00").unwrap();
        assert_eq!(next_day.seconds_since(&origin), 43_200);
        let next_month = Timestamp::parse("2026-09-01_12:00:00").unwrap();
        assert_eq!(next_month.seconds_since(&origin), 22 * 86_400);
    }
}
