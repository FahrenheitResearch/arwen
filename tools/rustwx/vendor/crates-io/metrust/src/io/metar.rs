//! METAR (aviation weather report) text parser.
//!
//! Regex-free implementation: splits the raw text on whitespace and
//! pattern-matches each token.

// ── Public types ────────────────────────────────────────────────────────

/// A parsed METAR observation.
#[derive(Debug, Clone)]
pub struct Metar {
    /// ICAO station identifier (e.g. "KATL").
    pub station: String,
    /// Observation time as "DDHHMMz".
    pub time: String,
    /// Wind direction in degrees (None for calm or variable).
    pub wind_direction: Option<f64>,
    /// Sustained wind speed in knots.
    pub wind_speed: Option<f64>,
    /// Gust speed in knots, if reported.
    pub wind_gust: Option<f64>,
    /// Prevailing visibility in statute miles.
    pub visibility: Option<f64>,
    /// Temperature in Celsius.
    pub temperature: Option<f64>,
    /// Dewpoint in Celsius.
    pub dewpoint: Option<f64>,
    /// Altimeter setting in inches of mercury.
    pub altimeter: Option<f64>,
    /// Sky condition layers: (cover_type, height_in_feet).
    /// Cover types: CLR, SKC, FEW, SCT, BKN, OVC, VV.
    pub sky_cover: Vec<(String, Option<u32>)>,
    /// Present weather phenomena tokens (e.g. "RA", "+TSRA", "-SN").
    pub weather: Vec<String>,
    /// Original raw METAR string.
    pub raw: String,
}

// ── Known weather phenomena prefixes / codes ────────────────────────────

/// Short weather descriptor/phenomenon codes used in METAR.
const WEATHER_PHENOMENA: &[&str] = &[
    "RA", "SN", "DZ", "SG", "IC", "PL", "GR", "GS", "UP",   // precipitation
    "FG", "BR", "HZ", "FU", "SA", "DU", "VA", "PY",          // obscuration
    "TS", "SH", "FZ", "BC", "MI", "PR", "BL", "DR",          // descriptors
    "SQ", "FC", "DS", "SS", "PO",                             // other
];

// ── Helpers ─────────────────────────────────────────────────────────────

/// Returns true if the token looks like a METAR weather phenomenon group.
fn is_weather_token(tok: &str) -> bool {
    if tok.is_empty() {
        return false;
    }
    // Strip leading intensity (+/-) and VC prefix.
    let mut s = tok;
    if s.starts_with('+') || s.starts_with('-') {
        s = &s[1..];
    }
    if s.starts_with("VC") {
        s = &s[2..];
    }
    if s.is_empty() {
        return false;
    }
    // The remainder should be composed entirely of 2-letter phenomenon codes.
    if s.len() % 2 != 0 {
        return false;
    }
    let mut i = 0;
    while i + 1 < s.len() {
        let code = &s[i..i + 2];
        if !WEATHER_PHENOMENA.contains(&code) {
            return false;
        }
        i += 2;
    }
    true
}

/// Returns true if `tok` looks like a sky-cover token (e.g. "SCT050", "OVC120", "CLR").
fn is_sky_token(tok: &str) -> bool {
    let prefixes = ["FEW", "SCT", "BKN", "OVC", "CLR", "SKC", "VV"];
    prefixes.iter().any(|p| tok.starts_with(p))
}

/// Parse a sky-cover token into (type, height_in_feet).
fn parse_sky(tok: &str) -> (String, Option<u32>) {
    let prefixes = ["FEW", "SCT", "BKN", "OVC", "CLR", "SKC", "VV"];
    for &p in &prefixes {
        if tok.starts_with(p) {
            let rest = &tok[p.len()..];
            // Strip optional CB/TCU suffix for height parsing.
            let height_str = rest.trim_end_matches("CB").trim_end_matches("TCU");
            let height = if height_str.is_empty() {
                None
            } else {
                // Height is in hundreds of feet (e.g., "050" -> 5000).
                height_str.parse::<u32>().ok().map(|h| h * 100)
            };
            return (p.to_string(), height);
        }
    }
    (tok.to_string(), None)
}

/// Parse a METAR temperature/dewpoint group like "25/18", "M03/M07", "25/".
fn parse_temp_dew(tok: &str) -> (Option<f64>, Option<f64>) {
    let parts: Vec<&str> = tok.splitn(2, '/').collect();
    let parse_one = |s: &str| -> Option<f64> {
        if s.is_empty() {
            return None;
        }
        if let Some(rest) = s.strip_prefix('M') {
            rest.parse::<f64>().ok().map(|v| -v)
        } else {
            s.parse::<f64>().ok()
        }
    };
    let temp = parts.first().and_then(|s| parse_one(s));
    let dew = parts.get(1).and_then(|s| parse_one(s));
    (temp, dew)
}

/// Try to parse visibility from one or two tokens.
/// Returns (vis_sm, tokens_consumed).
/// Handles: "10SM", "1/2SM", "1SM", "P6SM", "M1/4SM", "1 1/2SM".
fn parse_visibility(tokens: &[&str], idx: usize) -> (Option<f64>, usize) {
    let tok = tokens[idx];

    // Check if this token ends with SM.
    if tok.ends_with("SM") {
        let s = &tok[..tok.len() - 2];
        let s = s.strip_prefix('P').unwrap_or(s); // P6SM -> 6
        let s = s.strip_prefix('M').unwrap_or(s); // M1/4SM -> 1/4 (less than)
        if let Some(v) = parse_fraction(s) {
            return (Some(v), 1);
        }
    }

    // Check for compound like "1 1/2SM" — integer token followed by fraction+SM.
    if idx + 1 < tokens.len() {
        let next = tokens[idx + 1];
        if next.ends_with("SM") {
            if let Ok(whole) = tok.parse::<f64>() {
                let frac_str = &next[..next.len() - 2];
                if let Some(frac) = parse_fraction(frac_str) {
                    return (Some(whole + frac), 2);
                }
            }
        }
    }

    (None, 0)
}

/// Parse a fractional string like "1/2", "3/4", or a plain integer.
fn parse_fraction(s: &str) -> Option<f64> {
    if let Some((num, den)) = s.split_once('/') {
        let n = num.parse::<f64>().ok()?;
        let d = den.parse::<f64>().ok()?;
        if d == 0.0 {
            return None;
        }
        Some(n / d)
    } else {
        s.parse::<f64>().ok()
    }
}

// ── Public API ──────────────────────────────────────────────────────────

impl Metar {
    /// Parse a single METAR observation string.
    ///
    /// The input should be a single METAR line (may start with "METAR" or
    /// "SPECI").
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("Empty METAR string".into());
        }

        let mut tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.is_empty() {
            return Err("Empty METAR string".into());
        }

        // Skip leading "METAR" or "SPECI" prefix.
        if tokens[0] == "METAR" || tokens[0] == "SPECI" {
            tokens.remove(0);
        }
        if tokens.is_empty() {
            return Err("METAR has no content after type prefix".into());
        }

        // --- Station ID: 4 uppercase letters (ICAO) ---
        let station = tokens[0].to_string();

        // --- Time: DDHHMMz ---
        let time = if tokens.len() > 1 && tokens[1].ends_with('Z') {
            tokens[1].to_string()
        } else {
            String::new()
        };

        let mut wind_direction: Option<f64> = None;
        let mut wind_speed: Option<f64> = None;
        let mut wind_gust: Option<f64> = None;
        let mut visibility: Option<f64> = None;
        let mut temperature: Option<f64> = None;
        let mut dewpoint: Option<f64> = None;
        let mut altimeter: Option<f64> = None;
        let mut sky_cover: Vec<(String, Option<u32>)> = Vec::new();
        let mut weather: Vec<String> = Vec::new();

        let mut i = 2; // start after station + time
        while i < tokens.len() {
            let tok = tokens[i];

            // ── AUTO / COR — skip ──
            if tok == "AUTO" || tok == "COR" {
                i += 1;
                continue;
            }

            // ── Wind: dddssKT or dddssGggKT or VRBssKT ──
            if tok.ends_with("KT") || tok.ends_with("MPS") {
                let unit_len = if tok.ends_with("KT") { 2 } else { 3 };
                let w = &tok[..tok.len() - unit_len];
                if w.starts_with("VRB") {
                    wind_direction = None;
                    let rest = &w[3..];
                    if let Some(gi) = rest.find('G') {
                        wind_speed = rest[..gi].parse::<f64>().ok();
                        wind_gust = rest[gi + 1..].parse::<f64>().ok();
                    } else {
                        wind_speed = rest.parse::<f64>().ok();
                    }
                } else if w.len() >= 5 {
                    wind_direction = w[..3].parse::<f64>().ok();
                    let speed_part = &w[3..];
                    if let Some(gi) = speed_part.find('G') {
                        wind_speed = speed_part[..gi].parse::<f64>().ok();
                        wind_gust = speed_part[gi + 1..].parse::<f64>().ok();
                    } else {
                        wind_speed = speed_part.parse::<f64>().ok();
                    }
                }
                // Calm wind: 00000KT
                if tok == "00000KT" {
                    wind_direction = Some(0.0);
                    wind_speed = Some(0.0);
                }
                i += 1;
                continue;
            }

            // ── Visibility ──
            if tok.ends_with("SM") || (i + 1 < tokens.len() && tokens[i + 1].ends_with("SM")) {
                let (vis, consumed) = parse_visibility(&tokens, i);
                if consumed > 0 {
                    visibility = vis;
                    i += consumed;
                    continue;
                }
            }

            // ── Sky cover ──
            if is_sky_token(tok) {
                sky_cover.push(parse_sky(tok));
                i += 1;
                continue;
            }

            // ── Weather phenomena ──
            if is_weather_token(tok) {
                weather.push(tok.to_string());
                i += 1;
                continue;
            }

            // ── Temperature / Dewpoint (TT/TdTd) ──
            if tok.contains('/') && !tok.ends_with("SM") {
                // Heuristic: contains '/', parts are numeric or start with M.
                let parts: Vec<&str> = tok.splitn(2, '/').collect();
                let looks_like_temp = |s: &str| -> bool {
                    if s.is_empty() {
                        return true; // dewpoint can be missing
                    }
                    let s = s.strip_prefix('M').unwrap_or(s);
                    s.chars().all(|c| c.is_ascii_digit())
                };
                if parts.len() == 2 && looks_like_temp(parts[0]) && looks_like_temp(parts[1]) {
                    let (t, d) = parse_temp_dew(tok);
                    temperature = t;
                    dewpoint = d;
                    i += 1;
                    continue;
                }
            }

            // ── Altimeter (Axxxx) ──
            if tok.starts_with('A') && tok.len() == 5 {
                if let Ok(val) = tok[1..].parse::<f64>() {
                    altimeter = Some(val / 100.0);
                    i += 1;
                    continue;
                }
            }

            // ── RMK and beyond — stop parsing ──
            if tok == "RMK" {
                break;
            }

            i += 1;
        }

        Ok(Metar {
            station,
            time,
            wind_direction,
            wind_speed,
            wind_gust,
            visibility,
            temperature,
            dewpoint,
            altimeter,
            sky_cover,
            weather,
            raw: trimmed.to_string(),
        })
    }
}

/// Parse a multi-line file containing one METAR per line.
///
/// Blank lines and lines starting with `#` are skipped.  Returns all
/// successfully parsed METARs (silently ignoring unparseable lines).
pub fn parse_metar_file(content: &str) -> Vec<Metar> {
    content
        .lines()
        .filter(|line| {
            let t = line.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .filter_map(|line| Metar::parse(line).ok())
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_metar() {
        let raw = "KATL 121756Z 27015G25KT 10SM SCT050 BKN100 25/18 A2990 RMK AO2";
        let m = Metar::parse(raw).unwrap();
        assert_eq!(m.station, "KATL");
        assert_eq!(m.time, "121756Z");
        assert_eq!(m.wind_direction, Some(270.0));
        assert_eq!(m.wind_speed, Some(15.0));
        assert_eq!(m.wind_gust, Some(25.0));
        assert_eq!(m.visibility, Some(10.0));
        assert_eq!(m.temperature, Some(25.0));
        assert_eq!(m.dewpoint, Some(18.0));
        assert!((m.altimeter.unwrap() - 29.90).abs() < 0.001);
        assert_eq!(m.sky_cover.len(), 2);
        assert_eq!(m.sky_cover[0], ("SCT".to_string(), Some(5000)));
        assert_eq!(m.sky_cover[1], ("BKN".to_string(), Some(10000)));
    }

    #[test]
    fn metar_with_prefix() {
        let raw = "METAR KORD 121756Z 18010KT 6SM -RA OVC020 18/16 A2985";
        let m = Metar::parse(raw).unwrap();
        assert_eq!(m.station, "KORD");
        assert_eq!(m.wind_direction, Some(180.0));
        assert_eq!(m.wind_speed, Some(10.0));
        assert_eq!(m.wind_gust, None);
        assert_eq!(m.visibility, Some(6.0));
        assert_eq!(m.weather, vec!["-RA"]);
        assert_eq!(m.sky_cover.len(), 1);
        assert_eq!(m.sky_cover[0], ("OVC".to_string(), Some(2000)));
    }

    #[test]
    fn calm_wind() {
        let raw = "KJFK 121756Z 00000KT 10SM CLR 20/10 A3000";
        let m = Metar::parse(raw).unwrap();
        assert_eq!(m.wind_direction, Some(0.0));
        assert_eq!(m.wind_speed, Some(0.0));
        assert_eq!(m.wind_gust, None);
    }

    #[test]
    fn variable_wind() {
        let raw = "KLAX 121800Z VRB03KT 10SM FEW250 22/08 A2992";
        let m = Metar::parse(raw).unwrap();
        assert_eq!(m.wind_direction, None);
        assert_eq!(m.wind_speed, Some(3.0));
    }

    #[test]
    fn negative_temp() {
        let raw = "KDEN 121756Z 36010KT 10SM SKC M03/M07 A3020";
        let m = Metar::parse(raw).unwrap();
        assert_eq!(m.temperature, Some(-3.0));
        assert_eq!(m.dewpoint, Some(-7.0));
    }

    #[test]
    fn fractional_visibility() {
        let raw = "KIAH 121756Z 09005KT 1/2SM FG VV003 15/15 A3010";
        let m = Metar::parse(raw).unwrap();
        assert!((m.visibility.unwrap() - 0.5).abs() < 0.001);
        assert_eq!(m.weather, vec!["FG"]);
        assert_eq!(m.sky_cover.len(), 1);
        assert_eq!(m.sky_cover[0].0, "VV");
    }

    #[test]
    fn compound_visibility() {
        let raw = "KBOS 121756Z 27010KT 1 1/2SM BR SCT010 18/17 A2995";
        let m = Metar::parse(raw).unwrap();
        assert!((m.visibility.unwrap() - 1.5).abs() < 0.001);
    }

    #[test]
    fn thunderstorm_weather() {
        let raw = "KOKC 121756Z 23020G35KT 3SM +TSRA BKN030CB 28/22 A2970";
        let m = Metar::parse(raw).unwrap();
        assert_eq!(m.weather, vec!["+TSRA"]);
        assert_eq!(m.wind_gust, Some(35.0));
    }

    #[test]
    fn multiple_weather() {
        let raw = "KSFO 121756Z 28012KT 2SM -RA BR OVC010 14/13 A2998";
        let m = Metar::parse(raw).unwrap();
        assert_eq!(m.weather, vec!["-RA", "BR"]);
    }

    #[test]
    fn parse_file_multi_line() {
        let content = "\
# Header line
KATL 121756Z 27015KT 10SM SCT050 25/18 A2990
KORD 121756Z 18010KT 6SM OVC020 18/16 A2985

# Another comment
KJFK 121800Z 00000KT 10SM CLR 20/10 A3000
";
        let metars = parse_metar_file(content);
        assert_eq!(metars.len(), 3);
        assert_eq!(metars[0].station, "KATL");
        assert_eq!(metars[1].station, "KORD");
        assert_eq!(metars[2].station, "KJFK");
    }

    #[test]
    fn empty_and_error() {
        assert!(Metar::parse("").is_err());
        assert!(Metar::parse("   ").is_err());
    }

    #[test]
    fn is_weather_token_checks() {
        assert!(is_weather_token("RA"));
        assert!(is_weather_token("+TSRA"));
        assert!(is_weather_token("-SN"));
        assert!(is_weather_token("VCFG"));
        assert!(is_weather_token("FZRA"));
        assert!(!is_weather_token("KATL"));
        assert!(!is_weather_token("10SM"));
        assert!(!is_weather_token("A2990"));
        assert!(!is_weather_token(""));
    }

    #[test]
    fn parse_fraction_works() {
        assert!((parse_fraction("1/2").unwrap() - 0.5).abs() < 0.001);
        assert!((parse_fraction("3/4").unwrap() - 0.75).abs() < 0.001);
        assert!((parse_fraction("10").unwrap() - 10.0).abs() < 0.001);
        assert!(parse_fraction("abc").is_none());
    }

    #[test]
    fn p6sm_visibility() {
        let raw = "KDFW 121756Z 18005KT P6SM SKC 30/15 A2980";
        let m = Metar::parse(raw).unwrap();
        assert_eq!(m.visibility, Some(6.0));
    }
}
