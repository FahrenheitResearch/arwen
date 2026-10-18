//! IMS daily Northern Hemisphere snow and ice (NOAA/NESDIS IMS, distributed
//! by NSIDC as G02156 version 1.3), sampled at target points.
//!
//! Why this exists: a forcing analysis that does not resolve a lake cannot
//! say whether it is frozen.  On a GFS start the WPS nearest-water search
//! hands every such lake the ice (none) and skin temperature of the
//! nearest GFS water, so a frozen lake starts as open water far below
//! freezing.  IMS is an observed, public, keyless daily analysis of lake
//! and sea ice and of snow-covered ground at 1 km (4 km as well), and this
//! module is the reader the lake-ice provider samples it with
//! (`gpuwm/ingest/lake_ice.py`).
//!
//! The format read is the ASCII product (`imsYYYYDDD_00UTC_{1km,4km}_v1.3.asc.gz`):
//! a free-text header, then `N` rows of `N` characters `0`..`4`
//! (0 outside the hemisphere, 1 open water, 2 snow-free land, 3 sea or lake
//! ice, 4 snow-covered land), row 1 at the LOWER-LEFT corner of the polar
//! stereographic grid.  The header must say so, and must declare one of the
//! two grids this reader knows (6144 or 24576 square); anything else is
//! refused by name rather than sampled on a guessed geometry.
//!
//! Grid: polar stereographic, true at 60 N, straight-down longitude 80 W,
//! ellipsoid a = 6378137 m, b = 6356257 m, extent +-12,288,000 m in x and y
//! (the projection and coordinates NSIDC's NetCDF edition of the same product
//! declares: cell 1000 m on the 24576 grid, 4000 m on the 6144 grid).
//!
//! The gzip stream is inflated with the vendored `miniz_oxide`, and rows
//! are streamed: nothing past the last row a target needs is decompressed
//! and no row a target does not touch is kept.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use miniz_oxide::inflate::stream::{inflate, InflateState};
use miniz_oxide::{DataFormat, MZError, MZFlush, MZStatus};

/// Distinct IMS surface codes, 0..=4.
pub const IMS_CODES: usize = 5;
pub const IMS_SEMI_MAJOR_M: f64 = 6_378_137.0;
pub const IMS_SEMI_MINOR_M: f64 = 6_356_257.0;
pub const IMS_TRUE_LATITUDE_DEG: f64 = 60.0;
pub const IMS_CENTRAL_LONGITUDE_DEG: f64 = -80.0;
pub const IMS_HALF_EXTENT_M: f64 = 12_288_000.0;
/// The two grids of the v1.3 product this reader samples.
pub const IMS_DIMENSIONS: [usize; 2] = [6144, 24576];

#[derive(Debug)]
pub enum ImsError {
    Io(std::io::Error),
    Format(String),
}

impl std::fmt::Display for ImsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImsError::Io(error) => write!(f, "{error}"),
            ImsError::Format(message) => write!(f, "{message}"),
        }
    }
}

impl From<std::io::Error> for ImsError {
    fn from(error: std::io::Error) -> Self {
        ImsError::Io(error)
    }
}

fn format_error(message: impl Into<String>) -> ImsError {
    ImsError::Format(message.into())
}

/// Forward polar stereographic (north pole, ellipsoidal; Snyder 1987
/// eqs. 15-9, 14-15, 21-33, 21-34) onto the IMS plane, metres.
pub fn ims_xy(lat_deg: f64, lon_deg: f64) -> (f64, f64) {
    let a = IMS_SEMI_MAJOR_M;
    let e2 = 1.0 - (IMS_SEMI_MINOR_M * IMS_SEMI_MINOR_M) / (a * a);
    let e = libm::sqrt(e2);
    let quarter = std::f64::consts::FRAC_PI_4;
    let t = |phi: f64| {
        let s = libm::sin(phi);
        libm::tan(quarter - 0.5 * phi) / libm::pow((1.0 - e * s) / (1.0 + e * s), 0.5 * e)
    };
    let phi_c = IMS_TRUE_LATITUDE_DEG.to_radians();
    let s_c = libm::sin(phi_c);
    let m_c = libm::cos(phi_c) / libm::sqrt(1.0 - e2 * s_c * s_c);
    let rho = a * m_c * t(lat_deg.to_radians()) / t(phi_c);
    let dlam = (lon_deg - IMS_CENTRAL_LONGITUDE_DEG).to_radians();
    (rho * libm::sin(dlam), -rho * libm::cos(dlam))
}

/// The pixel (column, row-from-bottom) holding a point, or `None` off the grid.
pub fn ims_pixel(lat_deg: f64, lon_deg: f64, dimension: usize) -> Option<(usize, usize)> {
    if !(lat_deg.is_finite() && lon_deg.is_finite()) {
        return None;
    }
    let cell = 2.0 * IMS_HALF_EXTENT_M / dimension as f64;
    let (x, y) = ims_xy(lat_deg, lon_deg);
    let column = libm::floor((x + IMS_HALF_EXTENT_M) / cell);
    let row = libm::floor((y + IMS_HALF_EXTENT_M) / cell);
    if column < 0.0 || row < 0.0 || column >= dimension as f64 || row >= dimension as f64 {
        return None;
    }
    Some((column as usize, row as usize))
}

/// What one sampling pass saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImsSampleReceipt {
    /// The grid the header declared (6144 or 24576).
    pub dimension: usize,
    /// Data rows read before every target had its window.
    pub rows_read: usize,
    /// Targets whose centre fell off the grid (all counts zero, centre 0).
    pub off_grid: usize,
}

/// A gzip member's header length, or a refusal naming what is wrong.
fn gzip_header_len(bytes: &[u8]) -> Result<usize, ImsError> {
    if bytes.len() < 10 || bytes[0] != 0x1f || bytes[1] != 0x8b || bytes[2] != 8 {
        return Err(format_error("not a gzip (deflate) file"));
    }
    let flags = bytes[3];
    let mut at = 10usize;
    let need = |at: usize| -> Result<(), ImsError> {
        if at > bytes.len() {
            Err(format_error("gzip header runs past the bytes read"))
        } else {
            Ok(())
        }
    };
    if flags & 0x04 != 0 {
        need(at + 2)?;
        let extra = u16::from_le_bytes([bytes[at], bytes[at + 1]]) as usize;
        at += 2 + extra;
        need(at)?;
    }
    for flag in [0x08u8, 0x10u8] {
        if flags & flag != 0 {
            let end = bytes[at..]
                .iter()
                .position(|&b| b == 0)
                .ok_or_else(|| format_error("gzip header name or comment is unterminated"))?;
            at += end + 1;
        }
    }
    if flags & 0x02 != 0 {
        at += 2;
        need(at)?;
    }
    Ok(at)
}

/// Streams the inflated text of a `.gz` (or plain) file line by line.
struct LineStream {
    file: File,
    gz: Option<Box<InflateState>>,
    input: Vec<u8>,
    input_at: usize,
    input_end: usize,
    eof: bool,
    out: Vec<u8>,
    out_at: usize,
    out_end: usize,
    done: bool,
}

impl LineStream {
    fn open(path: &Path) -> Result<Self, ImsError> {
        let mut file = File::open(path)?;
        let mut input = vec![0u8; 1 << 20];
        let mut filled = 0usize;
        while filled < input.len() {
            let read = file.read(&mut input[filled..])?;
            if read == 0 {
                break;
            }
            filled += read;
        }
        let gzipped = filled >= 2 && input[0] == 0x1f && input[1] == 0x8b;
        let (gz, start) = if gzipped {
            (Some(InflateState::new_boxed(DataFormat::Raw)), gzip_header_len(&input[..filled])?)
        } else {
            (None, 0)
        };
        Ok(Self {
            file,
            gz,
            input,
            input_at: start,
            input_end: filled,
            eof: filled < (1 << 20),
            out: vec![0u8; 1 << 20],
            out_at: 0,
            out_end: 0,
            done: false,
        })
    }

    /// Refill `out`; false at the end of the stream.
    fn fill(&mut self) -> Result<bool, ImsError> {
        loop {
            if self.done {
                return Ok(false);
            }
            if self.input_at == self.input_end && !self.eof {
                let read = self.file.read(&mut self.input)?;
                self.input_at = 0;
                self.input_end = read;
                if read == 0 {
                    self.eof = true;
                }
            }
            match self.gz.as_mut() {
                None => {
                    if self.input_at == self.input_end {
                        self.done = true;
                        return Ok(false);
                    }
                    let count = self.input_end - self.input_at;
                    self.out[..count].copy_from_slice(&self.input[self.input_at..self.input_end]);
                    self.input_at = self.input_end;
                    self.out_at = 0;
                    self.out_end = count;
                    return Ok(true);
                }
                Some(state) => {
                    // Never MZFlush::Finish: on a first call it asks for the
                    // whole stream in one output buffer and fails otherwise.
                    let flush = MZFlush::None;
                    let result = inflate(
                        state,
                        &self.input[self.input_at..self.input_end],
                        &mut self.out,
                        flush,
                    );
                    self.input_at += result.bytes_consumed;
                    match result.status {
                        Ok(MZStatus::StreamEnd) => self.done = true,
                        Ok(_) => {}
                        Err(MZError::Buf) if !self.eof || result.bytes_written > 0 => {}
                        Err(error) => {
                            return Err(format_error(format!("the gzip stream is corrupt ({error:?})")));
                        }
                    }
                    if result.bytes_written > 0 {
                        self.out_at = 0;
                        self.out_end = result.bytes_written;
                        return Ok(true);
                    }
                    if self.eof && self.input_at == self.input_end && !self.done {
                        return Err(format_error("the gzip stream ends before its last block"));
                    }
                }
            }
        }
    }

    /// The next line without its terminator (`\n` or `\r\n`), into `line`.
    fn next_line(&mut self, line: &mut Vec<u8>) -> Result<bool, ImsError> {
        line.clear();
        loop {
            if self.out_at == self.out_end && !self.fill()? {
                return Ok(!line.is_empty());
            }
            let chunk = &self.out[self.out_at..self.out_end];
            if let Some(end) = chunk.iter().position(|&b| b == b'\n') {
                line.extend_from_slice(&chunk[..end]);
                self.out_at += end + 1;
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                return Ok(true);
            }
            line.extend_from_slice(chunk);
            self.out_at = self.out_end;
            if line.len() > 1 << 16 {
                return Err(format_error("a line is longer than any IMS row"));
            }
        }
    }
}

fn is_data_row(line: &[u8], dimension: usize) -> bool {
    line.len() == dimension && line.iter().all(|&b| (b'0'..=b'4').contains(&b))
}

/// Sample an IMS ASCII file at target points.
///
/// For target `t` with centre pixel `(c, r)`, every pixel in the square
/// window `c-radius..=c+radius`, `r-radius..=r+radius` (clipped to the
/// grid) is counted by code into `counts[t*5 + code]`, and the centre
/// pixel's own code goes to `centre[t]`.  A target off the grid keeps
/// zero counts and centre 0 (IMS's own "outside" code).
pub fn sample_ims_file(
    path: &Path,
    lat: &[f64],
    lon: &[f64],
    radius: usize,
    counts: &mut [u32],
    centre: &mut [u8],
) -> Result<ImsSampleReceipt, ImsError> {
    let n = lat.len();
    if lon.len() != n || counts.len() != n * IMS_CODES || centre.len() != n {
        return Err(format_error("target and output lengths differ"));
    }
    counts.iter_mut().for_each(|value| *value = 0);
    centre.iter_mut().for_each(|value| *value = 0);
    let mut stream = LineStream::open(path)?;
    let mut line = Vec::with_capacity(32_768);

    // Header: the grid and the row order, both declared in the text.
    let mut dimension = None;
    let mut lower_left = false;
    loop {
        if !stream.next_line(&mut line)? {
            return Err(format_error("the file ends before its first data row"));
        }
        let text = String::from_utf8_lossy(&line);
        let trimmed = text.trim();
        if let Some(rest) = trimmed.strip_prefix("Dimensions:") {
            let numbers: Vec<usize> = rest
                .split(|c: char| !c.is_ascii_digit())
                .filter(|part| !part.is_empty())
                .filter_map(|part| part.parse().ok())
                .collect();
            if numbers.len() != 2 || numbers[0] != numbers[1] {
                return Err(format_error(format!("the header declares a grid of {rest:?}, not a square one")));
            }
            if !IMS_DIMENSIONS.contains(&numbers[0]) {
                return Err(format_error(format!(
                    "the header declares a {0} x {0} grid; this reader knows the 6144 (4 km) and 24576 (1 km) grids only",
                    numbers[0]
                )));
            }
            if dimension.is_some_and(|d| d != numbers[0]) {
                return Err(format_error("the header declares two different grids"));
            }
            dimension = Some(numbers[0]);
        }
        if trimmed.to_ascii_lowercase().contains("starts at: lower left") {
            lower_left = true;
        }
        if let Some(d) = dimension {
            if is_data_row(&line, d) {
                break;
            }
        }
    }
    let dimension = dimension.unwrap();
    if !lower_left {
        return Err(format_error(
            "the header does not say its first row is the lower-left one, so the row order is unknown",
        ));
    }

    // Which targets each row feeds, in row order.
    let mut visits: Vec<(usize, usize)> = Vec::with_capacity(n * (2 * radius + 1));
    let mut columns = vec![0usize; n];
    let mut off_grid = 0usize;
    let mut centre_row = vec![usize::MAX; n];
    for t in 0..n {
        match ims_pixel(lat[t], lon[t], dimension) {
            None => off_grid += 1,
            Some((c, r)) => {
                columns[t] = c;
                centre_row[t] = r;
                let low = r.saturating_sub(radius);
                let high = (r + radius).min(dimension - 1);
                for row in low..=high {
                    visits.push((row, t));
                }
            }
        }
    }
    visits.sort_unstable();
    let last_row = visits.last().map(|v| v.0);

    let mut row = 0usize;
    let mut next = 0usize;
    let mut rows_read = 0usize;
    if let Some(last_row) = last_row {
        loop {
            if !is_data_row(&line, dimension) {
                return Err(format_error(format!(
                    "data row {} is not {dimension} characters of 0..4",
                    row + 1
                )));
            }
            rows_read = row + 1;
            while next < visits.len() && visits[next].0 == row {
                let t = visits[next].1;
                let c = columns[t];
                let low = c.saturating_sub(radius);
                let high = (c + radius).min(dimension - 1);
                for &code in &line[low..=high] {
                    counts[t * IMS_CODES + (code - b'0') as usize] += 1;
                }
                if centre_row[t] == row {
                    centre[t] = line[c] - b'0';
                }
                next += 1;
            }
            if row == last_row {
                break;
            }
            if !stream.next_line(&mut line)? {
                return Err(format_error(format!(
                    "the file ends after {} of {dimension} data rows",
                    row + 1
                )));
            }
            row += 1;
        }
    }
    Ok(ImsSampleReceipt { dimension, rows_read, off_grid })
}

/// C ABI: sample an IMS ASCII file (`.asc` or `.asc.gz`) at `n` targets.
///
/// `counts_out` receives `n * 5` u32 (per-code pixel counts in each
/// target's window), `centre_out` `n` bytes (the centre pixel's code), and
/// `receipt_out` three u64: the declared grid, data rows read, targets off
/// the grid.  A nonzero return leaves its sentence in
/// `gpuwm_bridge_last_error`.
///
/// # Safety
///
/// `path` must address `path_len` bytes of UTF-8; `lat` and `lon` `n`
/// readable f64 each; the outputs their stated writable sizes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_ims_sample_codes(
    path: *const u8,
    path_len: usize,
    lat: *const f64,
    lon: *const f64,
    n: usize,
    radius: u32,
    counts_out: *mut u32,
    centre_out: *mut u8,
    receipt_out: *mut u64,
) -> i32 {
    use crate::wif_ffi::{set_last_error, ERR_INPUT_FORMAT, ERR_INPUT_OPEN};
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if path.is_null() || path_len == 0 || receipt_out.is_null()
            || (n > 0 && (lat.is_null() || lon.is_null() || counts_out.is_null() || centre_out.is_null()))
        {
            set_last_error("a required pointer is null".to_string());
            return crate::ERR_NULL;
        }
        let Ok(text) = std::str::from_utf8(std::slice::from_raw_parts(path, path_len)) else {
            set_last_error("the IMS path is not valid UTF-8".to_string());
            return crate::ERR_NULL;
        };
        let file = Path::new(text);
        let (lat, lon, counts, centre) = if n == 0 {
            (&[][..], &[][..], &mut [][..], &mut [][..])
        } else {
            (
                std::slice::from_raw_parts(lat, n),
                std::slice::from_raw_parts(lon, n),
                std::slice::from_raw_parts_mut(counts_out, n * IMS_CODES),
                std::slice::from_raw_parts_mut(centre_out, n),
            )
        };
        match sample_ims_file(file, lat, lon, radius as usize, counts, centre) {
            Ok(receipt) => {
                let out = std::slice::from_raw_parts_mut(receipt_out, 3);
                out[0] = receipt.dimension as u64;
                out[1] = receipt.rows_read as u64;
                out[2] = receipt.off_grid as u64;
                crate::OK
            }
            Err(error) => {
                set_last_error(format!("{}: {error}", file.display()));
                match error {
                    ImsError::Io(_) => ERR_INPUT_OPEN,
                    ImsError::Format(_) => ERR_INPUT_FORMAT,
                }
            }
        }
    }))
    .unwrap_or(crate::ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn projection_matches_proj_reference_points() {
        // pyproj 3.7 with the product's own proj4 string
        // (+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-80 +a=6378137 +b=6356257).
        let (x, y) = ims_xy(90.0, 0.0);
        assert!(x.abs() < 1e-6 && y.abs() < 1e-6);
        let (x, y) = ims_xy(60.0, -80.0);
        assert!(x.abs() < 1e-6);
        // On the true-scale latitude the radius is a*m_c*t/t_c = a*m_c.
        let e2 = 1.0 - (IMS_SEMI_MINOR_M / IMS_SEMI_MAJOR_M).powi(2);
        let phi = 60f64.to_radians();
        let expected = IMS_SEMI_MAJOR_M * phi.cos() / (1.0 - e2 * phi.sin().powi(2)).sqrt();
        assert!((y + expected).abs() < 1e-6, "{y} vs {}", -expected);
        let (x, _) = ims_xy(60.0, 10.0);
        assert!((x - expected).abs() < 1e-6);
    }

    fn write_gz(path: &Path, text: &str) {
        let deflated = miniz_oxide::deflate::compress_to_vec(text.as_bytes(), 6);
        let mut file = File::create(path).unwrap();
        file.write_all(&[0x1f, 0x8b, 8, 0x08, 0, 0, 0, 0, 0, 3]).unwrap();
        file.write_all(b"ims_test.asc\0").unwrap();
        file.write_all(&deflated).unwrap();
        file.write_all(&[0; 8]).unwrap();
    }

    fn synthetic(dimension: usize, fill: impl Fn(usize, usize) -> u8) -> String {
        let mut text = String::from(
            "Format: I1\nDimensions: ",
        );
        text.push_str(&format!("{dimension} x {dimension}\n(1,1) starts at: lower left corner\n"));
        text.push_str(&format!("   Dimensions:   {dimension}   {dimension}\n   \n"));
        for row in 0..dimension {
            for column in 0..dimension {
                text.push((b'0' + fill(column, row)) as char);
            }
            text.push('\n');
        }
        text
    }

    #[test]
    fn rows_count_from_the_bottom_and_windows_count_by_code() {
        let dimension = 6144;
        let lat = [47.9, 47.9];
        let lon = [-91.5, -91.5];
        let (c, r) = ims_pixel(lat[0], lon[0], dimension).unwrap();
        // Ice at the centre, snow on its row's neighbours, water elsewhere.
        let text = synthetic(dimension, |column, row| {
            if column == c && row == r {
                3
            } else if row == r && column.abs_diff(c) == 1 {
                4
            } else {
                1
            }
        });
        let dir = std::env::temp_dir().join(format!("ims-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ims_test.asc.gz");
        write_gz(&path, &text);
        let mut counts = vec![0u32; 2 * IMS_CODES];
        let mut centre = vec![0u8; 2];
        let receipt = sample_ims_file(&path, &lat, &lon, 1, &mut counts, &mut centre).unwrap();
        assert_eq!(receipt.dimension, dimension);
        assert_eq!(receipt.rows_read, r + 2);
        assert_eq!(&counts[..5], &[0, 6, 0, 1, 2]);
        assert_eq!(centre, vec![3, 3]);
        let mut counts0 = vec![0u32; 2 * IMS_CODES];
        sample_ims_file(&path, &lat, &lon, 0, &mut counts0, &mut centre).unwrap();
        assert_eq!(&counts0[..5], &[0, 0, 0, 1, 0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_unknown_grid_is_refused_by_name() {
        let dir = std::env::temp_dir().join(format!("ims-test-grid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ims_bad.asc.gz");
        write_gz(&path, "Dimensions: 1024 x 1024\n(1,1) starts at: lower left corner\n");
        let error = sample_ims_file(&path, &[45.0], &[-90.0], 0, &mut [0; 5], &mut [0]).unwrap_err();
        assert!(error.to_string().contains("1024 x 1024"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
