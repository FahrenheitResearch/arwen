//! Fuzzy search for GRIB2 messages by human-readable query.
//!
//! Supports patterns like:
//! - `"temperature"` -- matches any temperature variable
//! - `"temperature 2m"` -- matches TMP at 2m above ground
//! - `"wind 10m"` -- matches UGRD/VGRD at 10m
//! - `"cape"` -- matches CAPE
//! - `"500mb height"` -- matches HGT at 500 mb

use super::parser::Grib2Message;
use super::tables;

/// Alias table: each entry maps a canonical name to a list of short aliases.
/// When a query term matches an alias, the canonical name is used for matching
/// against the parameter name from the GRIB2 tables.
const ALIASES: &[(&str, &[&str])] = &[
    ("temperature", &["temp", "tmp", "t2m"]),
    ("dewpoint", &["dpt", "td", "dewpt", "dew"]),
    ("wind", &["ugrd", "vgrd", "wnd", "u-component", "v-component"]),
    ("pressure", &["pres", "mslp", "prmsl"]),
    ("precipitation", &["precip", "apcp", "rain", "total precipitation"]),
    ("cape", &["convective available"]),
    ("cin", &["convective inhibition"]),
    ("reflectivity", &["refl", "refc", "dbz", "composite reflectivity"]),
    ("relative humidity", &["rh", "relhum"]),
    ("specific humidity", &["spfh"]),
    ("geopotential height", &["hgt", "height", "gph"]),
    ("visibility", &["vis"]),
    ("cloud", &["cld", "tcdc", "total cloud"]),
    ("snow", &["snod", "weasd", "snowfall"]),
    ("ice", &["icec", "icing"]),
    ("vorticity", &["vort", "absv"]),
    ("helicity", &["hlcy", "srh"]),
    ("lifted index", &["li", "lftx"]),
    ("convective inhibition", &["cin"]),
    ("precipitable water", &["pwat", "pw"]),
    ("surface", &["sfc"]),
    ("mean sea level", &["msl"]),
];

/// Level alias table: common shorthand for level descriptions.
const LEVEL_ALIASES: &[(&str, &[&str])] = &[
    ("2 m above ground", &["2m", "2 m", "2meter"]),
    ("10 m above ground", &["10m", "10 m", "10meter"]),
    ("surface", &["sfc", "ground"]),
    ("mean sea level", &["msl"]),
    ("entire atmosphere", &["atmosphere", "column"]),
    ("tropopause", &["trop"]),
    // Pressure levels: "500 mb" -> "500"
];

/// A search result with its relevance score.
struct ScoredResult<'a> {
    message: &'a Grib2Message,
    score: u32,
}

/// Search messages by human-readable query.
///
/// The query is split into whitespace-separated terms. Each term is matched
/// against the parameter name, level name, and level value of every message.
/// Results are ranked by relevance:
///   - Exact match of parameter name: 100 points
///   - Alias match (query term is an alias that expands to match): 80 points
///   - Substring match on parameter name: 60 points
///   - Level value match (e.g., "500" matches 500 mb): 40 points
///   - Level name substring match: 30 points
///   - Level alias match: 30 points
///
/// Only messages that match at least one query term are returned.
pub fn search_messages<'a>(messages: &'a [Grib2Message], query: &str) -> Vec<&'a Grib2Message> {
    let query_lower = query.to_lowercase();
    let terms: Vec<&str> = query_lower.split_whitespace().collect();

    if terms.is_empty() {
        return Vec::new();
    }

    let mut results: Vec<ScoredResult<'a>> = Vec::new();

    for msg in messages {
        let param_name = tables::parameter_name(
            msg.discipline,
            msg.product.parameter_category,
            msg.product.parameter_number,
        )
        .to_lowercase();

        let level_name = tables::level_name(msg.product.level_type).to_lowercase();
        let level_value = msg.product.level_value;

        // Build full level string like "500 isobaric surface" or "2 m above ground"
        let full_level = format!("{} {}", level_value, level_name);
        let full_level_lower = full_level.to_lowercase();

        let mut total_score: u32 = 0;
        let mut matched_terms = 0u32;

        for &term in &terms {
            let mut term_score: u32 = 0;

            // 1. Exact match on full parameter name
            if param_name == term {
                term_score = term_score.max(100);
            }

            // 2. Check if term is an alias -> expand to canonical, then match
            for &(canonical, aliases) in ALIASES {
                let term_is_alias = aliases.iter().any(|a| *a == term) || canonical == term;
                if term_is_alias {
                    // Check if canonical matches parameter name
                    if param_name.contains(canonical) {
                        term_score = term_score.max(80);
                    }
                    // Also check if any alias matches the parameter name directly
                    for &alias in aliases {
                        if param_name.contains(alias) {
                            term_score = term_score.max(80);
                        }
                    }
                }
            }

            // 3. Substring match on parameter name
            if param_name.contains(term) {
                term_score = term_score.max(60);
            }

            // 4. Level value match (e.g., "500" matches level_value 500.0)
            if let Ok(val) = term.parse::<f64>() {
                if (level_value - val).abs() < 0.5 {
                    term_score = term_score.max(40);
                }
            }

            // 5. Parse level shorthand like "500mb", "2m", "10m", "850hpa"
            let level_parsed = parse_level_term(term);
            if let Some((val, _unit)) = level_parsed {
                if (level_value - val).abs() < 0.5 {
                    term_score = term_score.max(40);
                }
            }

            // 6. Level name substring match
            if full_level_lower.contains(term) {
                term_score = term_score.max(30);
            }

            // 7. Level alias match
            for &(canonical_level, level_aliases) in LEVEL_ALIASES {
                if level_aliases.iter().any(|a| *a == term) {
                    if full_level_lower.contains(canonical_level)
                        || level_name.contains(canonical_level)
                    {
                        term_score = term_score.max(30);
                    }
                }
            }

            if term_score > 0 {
                matched_terms += 1;
                total_score += term_score;
            }
        }

        // Bonus for matching ALL query terms
        if matched_terms == terms.len() as u32 {
            total_score += 50;
        }

        if total_score > 0 {
            results.push(ScoredResult {
                message: msg,
                score: total_score,
            });
        }
    }

    // Sort by score descending
    results.sort_by(|a, b| b.score.cmp(&a.score));

    results.into_iter().map(|r| r.message).collect()
}

/// Parse level terms like "500mb", "2m", "10m", "850hpa" into (value, unit).
fn parse_level_term(term: &str) -> Option<(f64, &str)> {
    // Try common suffixes
    for suffix in &["mb", "hpa", "m"] {
        if term.ends_with(suffix) {
            let num_part = &term[..term.len() - suffix.len()];
            if let Ok(val) = num_part.parse::<f64>() {
                return Some((val, suffix));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grib2::parser::*;
    use chrono::NaiveDate;

    fn make_msg(discipline: u8, cat: u8, num: u8, level_type: u8, level_val: f64) -> Grib2Message {
        Grib2Message {
            discipline,
            reference_time: NaiveDate::from_ymd_opt(2024, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition::default(),
            product: ProductDefinition {
                parameter_category: cat,
                parameter_number: num,
                level_type,
                level_value: level_val,
                ..ProductDefinition::default()
            },
            data_rep: DataRepresentation::default(),
            bitmap: None,
            raw_data: Vec::new(),
        }
    }

    #[test]
    fn test_search_temperature() {
        let msgs = vec![
            make_msg(0, 0, 0, 103, 2.0),   // Temperature at 2m above ground
            make_msg(0, 0, 6, 103, 2.0),   // Dewpoint Temperature at 2m
            make_msg(0, 2, 2, 103, 10.0),  // U-Wind at 10m
        ];
        let results = search_messages(&msgs, "temperature");
        assert!(results.len() >= 1);
        // First result should be Temperature (exact match scores higher)
        let first_name = tables::parameter_name(
            results[0].discipline,
            results[0].product.parameter_category,
            results[0].product.parameter_number,
        );
        assert!(first_name.contains("Temperature"));
    }

    #[test]
    fn test_search_temperature_2m() {
        let msgs = vec![
            make_msg(0, 0, 0, 103, 2.0),     // Temperature at 2m above ground
            make_msg(0, 0, 0, 100, 500.0),    // Temperature at 500mb
            make_msg(0, 2, 2, 103, 10.0),     // U-Wind at 10m
        ];
        let results = search_messages(&msgs, "temperature 2m");
        assert!(!results.is_empty());
        // The 2m temperature should be the first result
        assert_eq!(results[0].product.level_value, 2.0);
        assert_eq!(results[0].product.level_type, 103);
    }

    #[test]
    fn test_search_wind_alias() {
        let msgs = vec![
            make_msg(0, 0, 0, 103, 2.0),   // Temperature at 2m
            make_msg(0, 2, 2, 103, 10.0),  // U-Component of Wind at 10m
            make_msg(0, 2, 3, 103, 10.0),  // V-Component of Wind at 10m
        ];
        let results = search_messages(&msgs, "wind 10m");
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_search_500mb_height() {
        let msgs = vec![
            make_msg(0, 3, 5, 100, 500.0),  // Geopotential Height at 500mb
            make_msg(0, 0, 0, 100, 500.0),  // Temperature at 500mb
            make_msg(0, 3, 5, 100, 850.0),  // Geopotential Height at 850mb
        ];
        let results = search_messages(&msgs, "500mb height");
        assert!(!results.is_empty());
        assert_eq!(results[0].product.parameter_category, 3);
        assert_eq!(results[0].product.parameter_number, 5);
        assert_eq!(results[0].product.level_value, 500.0);
    }

    #[test]
    fn test_search_cape() {
        let msgs = vec![
            make_msg(0, 7, 6, 1, 0.0),  // CAPE at surface
            make_msg(0, 0, 0, 103, 2.0), // Temperature at 2m
        ];
        let results = search_messages(&msgs, "cape");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_empty_query() {
        let msgs = vec![make_msg(0, 0, 0, 103, 2.0)];
        let results = search_messages(&msgs, "");
        assert!(results.is_empty());
    }

    #[test]
    fn test_no_match() {
        let msgs = vec![make_msg(0, 0, 0, 103, 2.0)];
        let results = search_messages(&msgs, "xyznonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_alias_rh() {
        let msgs = vec![
            make_msg(0, 1, 1, 103, 2.0),  // Relative Humidity at 2m
            make_msg(0, 0, 0, 103, 2.0),  // Temperature at 2m
        ];
        let results = search_messages(&msgs, "rh");
        assert!(!results.is_empty());
        let first_name = tables::parameter_name(
            results[0].discipline,
            results[0].product.parameter_category,
            results[0].product.parameter_number,
        );
        assert!(first_name.to_lowercase().contains("relative humidity"));
    }
}
