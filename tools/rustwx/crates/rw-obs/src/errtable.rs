//! GSI's conventional observation error table, read and interpolated the
//! way GSI reads and interpolates it.
//!
//! The table is data, not code: one block per report type, 33 pressure
//! levels, and per level the error of temperature, humidity, wind, surface
//! pressure and precipitable water.  A report type absent from the table,
//! or a variable a type does not carry, has GSI's fill `1e9`.  A new
//! table is a new file passed by path; the default is NCEP's published
//! table (`data/conventional-error-table.NOTICE.md` says which, with its
//! hash).
//!
//! **The read** (`converr.f90`, `converr_read`): a line `(1x,i3)` naming a
//! report type, then 33 lines `(1x,6e12.5)`, repeated to the end of the
//! file; a type named twice keeps its last block.
//!
//! **The interpolation** (`read_prepbufr.f90:1369-1400` of the HRRR
//! v4.1.21 GSI, the branch taken with `njqc = .false.`): the level's
//! pressure is clamped to 0..2000 hPa (a missing pressure, the library's
//! 1e11, becomes 2000); `k1` is 1 at or above the table's first pressure,
//! else the last table interval that brackets it, else 5 when it is at or
//! below the table's last pressure (GSI's own index, kept as it is); the
//! weight of level `k1 + 1` is clamped to 0..1.  GSI then floors the
//! result: temperature 0.5 K, humidity 0.05 tenths, wind 1 m/s, surface
//! pressure 0.3 hPa, precipitable water 1 mm (`read_prepbufr.f90:1099-1103`).

use std::error::Error;
use std::sync::OnceLock;

use crate::err;
use crate::ncep_bufr::BMISS;

/// GSI's table extents (`allocate(etabl(300,33,6))`).
pub const MAX_REPORT_TYPE: usize = 300;
pub const LEVELS: usize = 33;
pub const COLUMNS: usize = 6;
/// GSI's fill for a type or variable the table does not carry.
pub const FILL: f64 = 1.0e9;
/// GSI clamps the level's pressure to this before it looks it up.
pub const MAX_PRESSURE_HPA: f64 = 2000.0;

/// NCEP's published table, unchanged (see the NOTICE beside it).
pub const DEFAULT_TABLE_TEXT: &str = include_str!("../data/conventional-error-table.r3dv");
pub const DEFAULT_TABLE_NAME: &str = "conventional-error-table.r3dv";

/// A column of the table, and GSI's floor for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Column {
    Temperature,
    Humidity,
    Wind,
    SurfacePressure,
    PrecipitableWater,
}

impl Column {
    fn index(self) -> usize {
        match self {
            Column::Temperature => 1,
            Column::Humidity => 2,
            Column::Wind => 3,
            Column::SurfacePressure => 4,
            Column::PrecipitableWater => 5,
        }
    }

    /// GSI's lower limit (`terrmin`, `qerrmin`, `werrmin`, `perrmin`, `pwerrmin`).
    pub fn floor(self) -> f64 {
        match self {
            Column::Temperature => 0.5,
            Column::Humidity => 0.05,
            Column::Wind => 1.0,
            Column::SurfacePressure => 0.3,
            Column::PrecipitableWater => 1.0,
        }
    }
}

/// One table: per report type 1..=300, its 33 levels of six values.
#[derive(Debug, Clone)]
pub struct ErrorTable {
    blocks: Vec<Option<Box<[[f64; COLUMNS]; LEVELS]>>>,
    pub name: String,
    pub sha256: String,
    pub types: usize,
}

fn fortran_real(field: &str, what: &str) -> Result<f64, Box<dyn Error>> {
    let text = field.trim();
    // a blank Fortran field reads as zero
    if text.is_empty() {
        return Ok(0.0);
    }
    text.replace(['D', 'd'], "E")
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| err(format!("{what}: {text:?} is not a number")))
}

impl ErrorTable {
    /// Read a table in GSI's format.  Refused: a type outside 1..=300 (GSI's
    /// array would be indexed out of bounds), a block shorter than 33 lines,
    /// a field that is not a number, and a file with no block (GSI would run
    /// without a table).  A block whose pressures rise somewhere is read as
    /// it is: NCEP's own table has 500 where 600 belongs in types 152, 156,
    /// 158, 171 and 172, and GSI's search still finds a level for every
    /// pressure (the descending steps between the first and last pressure
    /// cover every value between them).
    pub fn parse(text: &str, name: &str) -> Result<Self, Box<dyn Error>> {
        let mut blocks: Vec<Option<Box<[[f64; COLUMNS]; LEVELS]>>> = vec![None; MAX_REPORT_TYPE + 1];
        let mut lines = text.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l)).enumerate().peekable();
        let mut types = 0usize;
        while let Some((n, line)) = lines.next() {
            if line.trim().is_empty() && lines.peek().map(|(_, l)| l.trim().is_empty()).unwrap_or(true) {
                break;
            }
            let head: String = line.chars().skip(1).take(3).collect();
            let report_type: usize = head
                .trim()
                .parse()
                .map_err(|_| err(format!("{name} line {}: {line:?} names no report type in columns 2 to 4", n + 1)))?;
            if !(1..=MAX_REPORT_TYPE).contains(&report_type) {
                return Err(err(format!("{name} line {}: report type {report_type} is outside 1..=300", n + 1)));
            }
            let mut block = Box::new([[0.0; COLUMNS]; LEVELS]);
            for level in block.iter_mut() {
                let (m, row) = lines
                    .next()
                    .ok_or_else(|| err(format!("{name}: the block of type {report_type} ends before its 33 levels")))?;
                let what = format!("{name} line {}", m + 1);
                let chars: Vec<char> = row.chars().collect();
                for (c, slot) in level.iter_mut().enumerate() {
                    let start = (1 + 12 * c).min(chars.len());
                    let end = (start + 12).min(chars.len());
                    *slot = fortran_real(&chars[start..end].iter().collect::<String>(), &what)?;
                }
            }
            blocks[report_type] = Some(block);
            types += 1;
        }
        if types == 0 {
            return Err(err(format!("{name} holds no report type")));
        }
        Ok(Self { blocks, name: name.to_string(), sha256: crate::hex_sha256(text.as_bytes()), types })
    }

    /// NCEP's published table, read once.
    pub fn default_table() -> &'static ErrorTable {
        static TABLE: OnceLock<ErrorTable> = OnceLock::new();
        TABLE.get_or_init(|| ErrorTable::parse(DEFAULT_TABLE_TEXT, DEFAULT_TABLE_NAME).expect("the vendored error table reads"))
    }

    pub fn has_type(&self, report_type: u16) -> bool {
        self.blocks.get(usize::from(report_type)).map(|b| b.is_some()).unwrap_or(false)
    }

    /// GSI's interpolated error for a report type at a level's pressure
    /// (hPa; `None` is missing), before the floor.  A type the table does not
    /// carry is GSI's fill.
    pub fn raw(&self, report_type: u16, pressure_hpa: Option<f64>, column: Column) -> f64 {
        let Some(block) = self.blocks.get(usize::from(report_type)).and_then(|b| b.as_deref()) else {
            return FILL;
        };
        let ppb = pressure_hpa.unwrap_or(BMISS).min(MAX_PRESSURE_HPA).max(0.0);
        let p = |k: usize| block[k - 1][0];
        let mut k1 = 1usize;
        if ppb >= p(1) {
            k1 = 1;
        }
        for kl in 1..LEVELS {
            if ppb >= p(kl + 1) && ppb <= p(kl) {
                k1 = kl;
            }
        }
        if ppb <= p(LEVELS) {
            k1 = 5;
        }
        let k2 = k1 + 1;
        let ediff = p(k2) - p(k1);
        let del = if ediff.abs() > f64::MIN_POSITIVE { (ppb - p(k1)) / ediff } else { f64::MAX };
        let del = del.min(1.0).max(0.0);
        let c = column.index();
        (1.0 - del) * block[k1 - 1][c] + del * block[k2 - 1][c]
    }

    /// The error GSI assigns: the interpolation, floored.
    pub fn error(&self, report_type: u16, pressure_hpa: Option<f64>, column: Column) -> f64 {
        self.raw(report_type, pressure_hpa, column).max(column.floor())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_table_reads_with_gsis_format() {
        let table = ErrorTable::default_table();
        assert_eq!(table.sha256, "9c5ddfd32751c93fb5e8689059f0e68d74360e5060b55d33035da90756f8b4fe");
        assert_eq!(table.types, 200);
        // values printed in the file for type 120 at 1100 hPa and type 181 at every level
        assert_eq!(table.raw(120, Some(1100.0), Column::Temperature), 1.2671);
        assert_eq!(table.raw(120, Some(1100.0), Column::SurfacePressure), 0.68115);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(close(table.raw(181, Some(612.3), Column::Humidity), 0.5912));
        assert!(close(table.raw(187, Some(980.0), Column::Temperature), 2.2585));
        assert!(close(table.raw(288, Some(1000.0), Column::Wind), 3.0));
        // a variable the type does not carry is the fill
        assert_eq!(table.raw(220, Some(500.0), Column::Temperature), FILL);
        assert!(table.has_type(224) && table.has_type(287));
        assert!(ErrorTable::parse(" 120 X\n", "t").unwrap_err().to_string().contains("ends before"));
    }

    fn block(values: impl Fn(usize) -> [f64; 6]) -> String {
        let mut text = String::from(" 130 OBSERVATION TYPE\n");
        for k in 0..LEVELS {
            let row = values(k);
            text.push(' ');
            for v in row {
                text.push_str(&format!("{v:12.5E}"));
            }
            text.push('\n');
        }
        text
    }

    #[test]
    fn the_interpolation_is_gsis_including_its_index_past_the_last_level() {
        // pressures 1100, 1050, ..., then 0 at level 33; temperature error = level number
        let pressures: Vec<f64> = (0..LEVELS).map(|k| if k == LEVELS - 1 { 0.0 } else { 1100.0 - 50.0 * k as f64 }).collect();
        let text = block(|k| [pressures[k], (k + 1) as f64, FILL, 2.0, 0.1, FILL]);
        let table = ErrorTable::parse(&text, "test").unwrap();
        // halfway between 1000 hPa (level 3) and 950 hPa (level 4)
        assert!((table.raw(130, Some(975.0), Column::Temperature) - 3.5).abs() < 1e-12);
        // above the first pressure: level 1; a missing pressure is 2000 hPa
        assert_eq!(table.raw(130, Some(1200.0), Column::Temperature), 1.0);
        assert_eq!(table.raw(130, None, Column::Temperature), 1.0);
        // at the last pressure GSI takes k1 = 5: the weight clamps to level 6
        assert_eq!(table.raw(130, Some(0.0), Column::Temperature), 6.0);
        // the floor
        assert_eq!(table.error(130, Some(975.0), Column::SurfacePressure), 0.3);
        assert_eq!(table.error(130, Some(975.0), Column::Wind), 2.0);
        // a type the table does not carry is the fill
        assert_eq!(table.raw(131, Some(500.0), Column::Temperature), FILL);
        // a block whose pressures rise in one place (as NCEP's types 152 to 172 do) still finds a level
        let mut kinked = pressures.clone();
        kinked[10] = 500.0;
        kinked[11] = 550.0;
        let text = block(|k| [kinked[k], (k + 1) as f64, FILL, 2.0, 0.1, FILL]);
        let table = ErrorTable::parse(&text, "kinked").unwrap();
        // 600 hPa lies only in the step from level 10 (650) to level 11 (500)
        assert!((table.raw(130, Some(600.0), Column::Temperature) - (10.0 + 50.0 / 150.0)).abs() < 1e-12);
        // refusals name what they prevent
        assert!(ErrorTable::parse(" 301 X\n", "t").unwrap_err().to_string().contains("outside"));
        assert!(ErrorTable::parse("", "t").unwrap_err().to_string().contains("no report type"));
    }
}
