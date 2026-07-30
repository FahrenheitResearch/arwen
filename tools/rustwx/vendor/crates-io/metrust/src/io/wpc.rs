//! WPC Coded Surface Bulletin parser.
//!
//! Parses NWS Weather Prediction Center (WPC) coded surface analysis
//! bulletins, extracting pressure centers (HIGH/LOW) and frontal
//! boundaries (WARM, COLD, STNRY, OCFNT, TROF) with their positions.
//!
//! Reference: MetPy's `metpy.io.text.parse_wpc_surface_bulletin`.

// ── Public types ────────────────────────────────────────────────────────

/// A single decoded feature from a WPC coded surface bulletin.
#[derive(Debug, Clone)]
pub struct SurfaceBulletinFeature {
    /// Feature type: `"HIGH"`, `"LOW"`, `"WARM"`, `"COLD"`, `"STNRY"`,
    /// `"OCFNT"`, or `"TROF"`.
    pub feature_type: String,
    /// Lat/lon pairs describing the feature location.
    /// For pressure centers this is a single point; for fronts it is a
    /// polyline of two or more points.
    pub points: Vec<(f64, f64)>,
    /// Central pressure (mb) for HIGH/LOW features, or `None` for fronts.
    pub value: Option<f64>,
    /// Strength qualifier for fronts (e.g. "WK", "MDT", "STG"), if present.
    pub strength: Option<String>,
    /// Valid time string parsed from the bulletin (e.g. "03121200").
    pub valid_time: Option<String>,
}

// ── Coordinate decoding ─────────────────────────────────────────────────

/// Decode a WPC coordinate string into a (lat, lon) tuple.
///
/// In the WPC coded surface bulletin, latitude and longitude are given in
/// degrees north and degrees west respectively. This function always
/// returns longitude as negative (west).
///
/// Hi-res bulletins use 7 digits; regular bulletins use 4 or 5 digits.
/// A leading `-` indicates southern hemisphere.
fn decode_coords(s: &str) -> Option<(f64, f64)> {
    let (coords, flip) = if let Some(stripped) = s.strip_prefix('-') {
        (stripped, -1.0)
    } else {
        (s, 1.0)
    };

    // All characters must be digits
    if coords.is_empty() || !coords.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    let split_pos = coords.len() / 2;
    if split_pos < 2 {
        return None;
    }

    let lat_str = &coords[..split_pos];
    let lon_str = &coords[split_pos..];

    if lat_str.len() < 2 || lon_str.len() < 3 {
        return None;
    }

    // Insert decimal point: first 2 digits of lat are integer part,
    // first 3 digits of lon are integer part
    let lat_int = &lat_str[..2];
    let lat_frac = &lat_str[2..];
    let lon_int = &lon_str[..3];
    let lon_frac = &lon_str[3..];

    let lat_s = if lat_frac.is_empty() {
        lat_int.to_string()
    } else {
        format!("{}.{}", lat_int, lat_frac)
    };
    let lon_s = if lon_frac.is_empty() {
        lon_int.to_string()
    } else {
        format!("{}.{}", lon_int, lon_frac)
    };

    let lat: f64 = lat_s.parse().ok()?;
    let lon: f64 = lon_s.parse().ok()?;

    Some((lat * flip, -lon))
}

/// Check if a token looks like a pressure value (3-4 digit number
/// starting with 8, 9, or 1).
fn is_pressure(s: &str) -> bool {
    if s.len() > 4 || s.is_empty() {
        return false;
    }
    let first = s.as_bytes()[0];
    (first == b'8' || first == b'9' || first == b'1') && s.chars().all(|c| c.is_ascii_digit())
}

/// Check if a token is a strength qualifier.
fn is_strength(s: &str) -> bool {
    matches!(s, "WK" | "MDT" | "STG")
}

/// Check if a token looks like a coordinate string (all digits, 4-8
/// chars, or starting with `-` followed by digits).
fn is_coord_token(s: &str) -> bool {
    let core = s.strip_prefix('-').unwrap_or(s);
    if core.is_empty() {
        return false;
    }
    let len = core.len();
    (4..=8).contains(&len) && core.chars().all(|c| c.is_ascii_digit())
}

// ── Line regrouping ─────────────────────────────────────────────────────

/// Regroup continuation lines: lines that start with a digit are
/// appended to the previous logical line.
fn regroup_lines(text: &str) -> Vec<Vec<String>> {
    let mut result: Vec<Vec<String>> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<String> = trimmed.split_whitespace().map(|s| s.to_string()).collect();
        if parts.is_empty() {
            continue;
        }
        // If the line starts with a digit and we have a previous group, append
        let first_char = parts[0].as_bytes()[0];
        if first_char.is_ascii_digit() && !result.is_empty() {
            result.last_mut().unwrap().extend(parts);
        } else {
            result.push(parts);
        }
    }

    result
}

// ── Main parser ─────────────────────────────────────────────────────────

/// Parse a WPC coded surface analysis bulletin and return a list of
/// decoded features (pressure centers and frontal boundaries).
///
/// The input `text` should be the full bulletin text. The parser extracts
/// VALID time lines, HIGHS/LOWS with pressure values and positions, and
/// front types (WARM, COLD, STNRY, OCFNT, TROF) with polyline positions.
pub fn parse_wpc_surface_bulletin(text: &str) -> Vec<SurfaceBulletinFeature> {
    let mut features = Vec::new();
    let mut valid_time: Option<String> = None;

    for parts in regroup_lines(text) {
        if parts.is_empty() {
            continue;
        }

        let keyword = parts[0].as_str();

        // Check for VALID time line
        if keyword == "VALID" || (keyword == "SURFACE" && parts.len() > 2 && parts[2] == "VALID")
        {
            // The time string is the last token
            if let Some(ts) = parts.last() {
                valid_time = Some(ts.clone());
            }
            continue;
        }

        match keyword {
            "HIGHS" | "LOWS" => {
                let feature_type = if keyword == "HIGHS" {
                    "HIGH"
                } else {
                    "LOW"
                };
                let info = &parts[1..];
                let mut current_pressure: Option<f64> = None;

                for token in info {
                    if is_pressure(token) {
                        current_pressure = token.parse::<f64>().ok();
                    } else if is_coord_token(token) {
                        if let Some((lat, lon)) = decode_coords(token) {
                            features.push(SurfaceBulletinFeature {
                                feature_type: feature_type.to_string(),
                                points: vec![(lat, lon)],
                                value: current_pressure,
                                strength: None,
                                valid_time: valid_time.clone(),
                            });
                        }
                    }
                }
            }
            "WARM" | "COLD" | "STNRY" | "OCFNT" | "TROF" => {
                let info = &parts[1..];
                let (strength, boundary_tokens) = if !info.is_empty() && is_strength(&info[0]) {
                    (Some(info[0].clone()), &info[1..])
                } else {
                    (None, info)
                };

                let points: Vec<(f64, f64)> = boundary_tokens
                    .iter()
                    .filter(|t| is_coord_token(t))
                    .filter_map(|t| decode_coords(t))
                    .collect();

                if !points.is_empty() {
                    features.push(SurfaceBulletinFeature {
                        feature_type: keyword.to_string(),
                        points,
                        value: None,
                        strength,
                        valid_time: valid_time.clone(),
                    });
                }
            }
            _ => {}
        }
    }

    features
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_coords_7digit() {
        // "4731193" -> lat=47.3, lon=-119.3
        let (lat, lon) = decode_coords("4731193").unwrap();
        assert!((lat - 47.3).abs() < 0.01);
        assert!((lon - (-119.3)).abs() < 0.01);
    }

    #[test]
    fn test_decode_coords_5digit() {
        // "47119" -> lat=47, lon=-119 (no fractional parts beyond split)
        // split_pos = 2, lat_str="47", lon_str="119"
        let (lat, lon) = decode_coords("47119").unwrap();
        assert!((lat - 47.0).abs() < 0.01);
        assert!((lon - (-119.0)).abs() < 0.01);
    }

    #[test]
    fn test_decode_coords_southern_hemisphere() {
        let (lat, lon) = decode_coords("-4731193").unwrap();
        assert!((lat - (-47.3)).abs() < 0.01);
        assert!((lon - (-119.3)).abs() < 0.01);
    }

    #[test]
    fn test_decode_coords_invalid() {
        assert!(decode_coords("abc").is_none());
        assert!(decode_coords("12").is_none());
        assert!(decode_coords("").is_none());
    }

    #[test]
    fn test_is_pressure() {
        assert!(is_pressure("1013"));
        assert!(is_pressure("999"));
        assert!(is_pressure("980"));
        assert!(!is_pressure("COLD"));
        assert!(!is_pressure("47119"));
        assert!(!is_pressure(""));
    }

    #[test]
    fn test_parse_highs_and_lows() {
        let bulletin = "\
VALID 03121200
HIGHS 1030 4731193 1025 3509700
LOWS 998 4510000 1005 5012000
";
        let features = parse_wpc_surface_bulletin(bulletin);
        assert_eq!(features.len(), 4);

        assert_eq!(features[0].feature_type, "HIGH");
        assert!((features[0].value.unwrap() - 1030.0).abs() < 0.1);
        assert_eq!(features[0].points.len(), 1);

        assert_eq!(features[1].feature_type, "HIGH");
        assert!((features[1].value.unwrap() - 1025.0).abs() < 0.1);

        assert_eq!(features[2].feature_type, "LOW");
        assert!((features[2].value.unwrap() - 998.0).abs() < 0.1);

        assert_eq!(features[3].feature_type, "LOW");
        assert!((features[3].value.unwrap() - 1005.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_fronts() {
        let bulletin = "\
VALID 03121200
COLD 4731193 4531093 4330993
WARM WK 3509700 3609800
TROF 4010000 4210100
";
        let features = parse_wpc_surface_bulletin(bulletin);
        assert_eq!(features.len(), 3);

        assert_eq!(features[0].feature_type, "COLD");
        assert_eq!(features[0].points.len(), 3);
        assert!(features[0].strength.is_none());
        assert!(features[0].value.is_none());

        assert_eq!(features[1].feature_type, "WARM");
        assert_eq!(features[1].points.len(), 2);
        assert_eq!(features[1].strength.as_deref(), Some("WK"));

        assert_eq!(features[2].feature_type, "TROF");
        assert_eq!(features[2].points.len(), 2);
    }

    #[test]
    fn test_valid_time_propagation() {
        let bulletin = "\
VALID 03121200
HIGHS 1030 4731193
VALID 03121800
LOWS 998 4510000
";
        let features = parse_wpc_surface_bulletin(bulletin);
        assert_eq!(features[0].valid_time.as_deref(), Some("03121200"));
        assert_eq!(features[1].valid_time.as_deref(), Some("03121800"));
    }

    #[test]
    fn test_continuation_lines() {
        let bulletin = "\
VALID 03121200
COLD 4731193
4531093 4330993
";
        let features = parse_wpc_surface_bulletin(bulletin);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].feature_type, "COLD");
        assert_eq!(features[0].points.len(), 3);
    }

    #[test]
    fn test_empty_bulletin() {
        let features = parse_wpc_surface_bulletin("");
        assert!(features.is_empty());
    }

    #[test]
    fn test_stnry_and_ocfnt() {
        let bulletin = "\
VALID 03121200
STNRY STG 4010000 4110100 4210200
OCFNT 3509700 3609800 3709900
";
        let features = parse_wpc_surface_bulletin(bulletin);
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].feature_type, "STNRY");
        assert_eq!(features[0].strength.as_deref(), Some("STG"));
        assert_eq!(features[0].points.len(), 3);
        assert_eq!(features[1].feature_type, "OCFNT");
        assert_eq!(features[1].points.len(), 3);
    }
}
