/// A single entry from a GRIB2 .idx index file.
///
/// Format: `msg_num:byte_offset:d=YYYYMMDDHH:variable:level:forecast_time:`
///
/// Extended fields are parsed from additional .idx columns when present:
/// - `ENS=+N` → ensemble_member
/// - `0-3 hr acc fcst` → statistical_process + time_range
#[derive(Debug, Clone)]
pub struct IdxEntry {
    pub msg_num: u32,
    pub byte_offset: u64,
    pub date: String,
    pub variable: String,
    pub level: String,
    pub forecast: String,
    /// Ensemble member number (e.g., from `ENS=+5`).
    pub ensemble_member: Option<u32>,
    /// Statistical process type: "acc", "avg", "max", "min".
    pub statistical_process: Option<String>,
    /// Time range as (start_fhour, end_fhour) for accumulated/averaged fields.
    pub time_range: Option<(u32, u32)>,
}

/// Search criteria for structured inventory queries.
///
/// All fields are optional; only non-None fields are checked. An entry must
/// match every specified criterion to be included in results.
#[derive(Debug, Clone, Default)]
pub struct SearchCriteria {
    /// Exact variable name match (e.g., "TMP", "APCP").
    pub variable: Option<String>,
    /// Substring match on level string (e.g., "500 mb", "surface").
    pub level: Option<String>,
    /// Level type category: "surface", "pressure", "height", "atmosphere".
    pub level_type: Option<String>,
    /// Forecast hour filter — matches entries whose forecast field contains this hour.
    pub forecast_hour: Option<u32>,
    /// Statistical process filter: "acc", "avg", "max", "min", "inst".
    pub statistical: Option<String>,
    /// Ensemble member number filter.
    pub ensemble: Option<u32>,
}

/// Parse the text content of a GRIB2 .idx file into a list of entries.
///
/// Handles both standard 6-field and extended .idx formats. Parses ensemble
/// member info from `ENS=+N` tokens and statistical process / time range
/// from forecast strings like `"0-3 hr acc fcst"`.
pub fn parse_idx(text: &str) -> Vec<IdxEntry> {
    let mut entries = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Split into at most 8 parts to capture extended fields
        let parts: Vec<&str> = line.splitn(8, ':').collect();
        if parts.len() < 6 {
            continue;
        }
        let msg_num = match parts[0].parse::<u32>() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let byte_offset = match parts[1].parse::<u64>() {
            Ok(n) => n,
            Err(_) => continue,
        };
        // parts[2] is like "d=2026031012"
        let date = parts[2]
            .strip_prefix("d=")
            .unwrap_or(parts[2])
            .to_string();
        let variable = parts[3].to_string();
        let level = parts[4].to_string();
        let forecast = parts[5].trim_end_matches(':').to_string();

        // Parse extended fields from remaining parts and the forecast string
        let mut ensemble_member = None;
        let mut statistical_process = None;
        // Check for ENS=+N in extra fields (parts[6], parts[7], ...)
        for i in 6..parts.len() {
            let extra = parts[i].trim().trim_end_matches(':');
            if let Some(ens_str) = extra.strip_prefix("ENS=") {
                // ENS=+5 or ENS=-3 or ENS=5
                let num_str = ens_str.trim_start_matches('+').trim_start_matches('-');
                if let Ok(n) = num_str.parse::<u32>() {
                    ensemble_member = Some(n);
                }
            }
        }

        // Parse statistical process and time range from forecast string
        // Patterns: "0-3 hr acc fcst", "6 hr avg fcst", "0-6 hr max fcst", "anl"
        let forecast_lower = forecast.to_lowercase();
        if forecast_lower.contains(" acc ") || forecast_lower.ends_with(" acc") {
            statistical_process = Some("acc".to_string());
        } else if forecast_lower.contains(" avg ") || forecast_lower.ends_with(" avg") {
            statistical_process = Some("avg".to_string());
        } else if forecast_lower.contains(" max ") || forecast_lower.ends_with(" max") {
            statistical_process = Some("max".to_string());
        } else if forecast_lower.contains(" min ") || forecast_lower.ends_with(" min") {
            statistical_process = Some("min".to_string());
        }

        // Parse time range from "N-M hr" pattern
        let time_range = parse_time_range(&forecast);

        entries.push(IdxEntry {
            msg_num,
            byte_offset,
            date,
            variable,
            level,
            forecast,
            ensemble_member,
            statistical_process,
            time_range,
        });
    }
    entries
}

/// Parse a time range like "0-3 hr" from a forecast string.
///
/// Returns `Some((start, end))` for patterns like `"0-3 hr acc fcst"`,
/// or `None` if no range is found.
fn parse_time_range(forecast: &str) -> Option<(u32, u32)> {
    // Look for pattern: digits-digits followed by " hr"
    let parts: Vec<&str> = forecast.split_whitespace().collect();
    for (i, part) in parts.iter().enumerate() {
        if part.contains('-') && i + 1 < parts.len() && parts[i + 1].starts_with("hr") {
            let range_parts: Vec<&str> = part.split('-').collect();
            if range_parts.len() == 2 {
                if let (Ok(start), Ok(end)) = (
                    range_parts[0].parse::<u32>(),
                    range_parts[1].parse::<u32>(),
                ) {
                    return Some((start, end));
                }
            }
        }
    }
    None
}

/// Find entries matching a search pattern.
///
/// Pattern format: `"VAR:level"` — matches variable name exactly and level as a substring.
/// If the pattern contains no colon, it matches only the variable name.
///
/// Examples:
/// - `"TMP:2 m above ground"` matches TMP at 2m
/// - `"CAPE:surface"` matches surface CAPE
/// - `"REFC"` matches any REFC entry
/// - `"MXUPHL"` matches any max updraft helicity entry
pub fn find_entries<'a>(entries: &'a [IdxEntry], pattern: &str) -> Vec<&'a IdxEntry> {
    let (var_pat, level_pat) = if let Some(idx) = pattern.find(':') {
        (&pattern[..idx], Some(&pattern[idx + 1..]))
    } else {
        (pattern, None)
    };

    entries
        .iter()
        .filter(|e| {
            if e.variable != var_pat {
                return false;
            }
            if let Some(lp) = level_pat {
                e.level.contains(lp)
            } else {
                true
            }
        })
        .collect()
}

/// Find entries matching a regex pattern.
///
/// The pattern is matched against a reconstructed .idx line:
/// `"variable:level:forecast"`. This allows flexible matching across
/// all fields simultaneously.
///
/// Examples:
/// - `"TMP.*surface"` — temperature at surface
/// - `"(?i)apcp.*acc"` — accumulated precipitation (case-insensitive)
/// - `"(UGRD|VGRD):10 m"` — wind components at 10m
pub fn find_entries_regex<'a>(entries: &'a [IdxEntry], pattern: &str) -> Vec<&'a IdxEntry> {
    // Use a simple regex-like matching without pulling in the regex crate.
    // We implement common patterns: "|" for alternation, ".*" for wildcard,
    // and "(?i)" for case-insensitive.
    let case_insensitive = pattern.starts_with("(?i)");
    let pat = if case_insensitive {
        &pattern[4..]
    } else {
        pattern
    };

    // Check if this is an alternation pattern like "(A|B|C)"
    if pat.starts_with('(') && pat.contains('|') {
        // Find the closing paren
        if let Some(close) = pat.find(')') {
            let alternatives: Vec<&str> = pat[1..close].split('|').collect();
            let rest = &pat[close + 1..];

            return entries
                .iter()
                .filter(|e| {
                    let line = format!("{}:{}:{}", e.variable, e.level, e.forecast);
                    let line_match = if case_insensitive {
                        line.to_lowercase()
                    } else {
                        line.clone()
                    };

                    for alt in &alternatives {
                        let alt_match = if case_insensitive {
                            alt.to_lowercase()
                        } else {
                            alt.to_string()
                        };
                        if line_match.contains(&alt_match) {
                            // Also check the rest of the pattern after the alternation
                            if rest.is_empty() {
                                return true;
                            }
                            let rest_clean = rest.trim_start_matches(".*").trim_start_matches(':');
                            if rest_clean.is_empty() {
                                return true;
                            }
                            let rest_match = if case_insensitive {
                                rest_clean.to_lowercase()
                            } else {
                                rest_clean.to_string()
                            };
                            if line_match.contains(&rest_match) {
                                return true;
                            }
                        }
                    }
                    false
                })
                .collect();
        }
    }

    // Simple wildcard matching: split on ".*" and check all parts exist in order
    let parts: Vec<&str> = pat.split(".*").collect();

    entries
        .iter()
        .filter(|e| {
            let line = format!("{}:{}:{}", e.variable, e.level, e.forecast);
            let line_match = if case_insensitive {
                line.to_lowercase()
            } else {
                line
            };

            let mut search_from = 0usize;
            for part in &parts {
                if part.is_empty() {
                    continue;
                }
                let part_match = if case_insensitive {
                    part.to_lowercase()
                } else {
                    part.to_string()
                };
                if let Some(pos) = line_match[search_from..].find(&part_match) {
                    search_from += pos + part_match.len();
                } else {
                    return false;
                }
            }
            true
        })
        .collect()
}

/// Find entries matching structured search criteria.
///
/// All specified criteria must match for an entry to be included. `None`
/// fields are ignored (match anything).
///
/// # Level type matching
///
/// The `level_type` field matches categories of levels:
/// - `"surface"` → level string contains "surface"
/// - `"pressure"` → level string contains "mb"
/// - `"height"` → level string contains "m above ground"
/// - `"atmosphere"` → level string contains "entire atmosphere"
pub fn find_entries_criteria<'a>(
    entries: &'a [IdxEntry],
    criteria: &SearchCriteria,
) -> Vec<&'a IdxEntry> {
    entries
        .iter()
        .filter(|e| {
            // Variable filter (exact match)
            if let Some(ref var) = criteria.variable {
                if e.variable != *var {
                    return false;
                }
            }

            // Level substring filter
            if let Some(ref level) = criteria.level {
                if !e.level.contains(level.as_str()) {
                    return false;
                }
            }

            // Level type category filter
            if let Some(ref lt) = criteria.level_type {
                let matches = match lt.as_str() {
                    "surface" => e.level.contains("surface"),
                    "pressure" => e.level.contains(" mb"),
                    "height" => e.level.contains("m above ground"),
                    "atmosphere" | "column" => {
                        e.level.contains("entire atmosphere")
                            || e.level.contains("entire column")
                    }
                    "tropopause" => e.level.contains("tropopause"),
                    "cloud" => e.level.contains("cloud"),
                    _ => e.level.to_lowercase().contains(&lt.to_lowercase()),
                };
                if !matches {
                    return false;
                }
            }

            // Forecast hour filter
            if let Some(fh) = criteria.forecast_hour {
                let fh_str = format!("{} hr fcst", fh);
                let fh_str2 = format!("{}hr fcst", fh);
                let anl_match = fh == 0
                    && (e.forecast == "anl"
                        || e.forecast.contains("analysis"));
                if !e.forecast.contains(&fh_str)
                    && !e.forecast.contains(&fh_str2)
                    && !anl_match
                {
                    return false;
                }
            }

            // Statistical process filter
            if let Some(ref stat) = criteria.statistical {
                match stat.as_str() {
                    "inst" | "instantaneous" => {
                        // Instantaneous means NO statistical process
                        if e.statistical_process.is_some() {
                            return false;
                        }
                    }
                    _ => {
                        if e.statistical_process.as_deref() != Some(stat.as_str()) {
                            return false;
                        }
                    }
                }
            }

            // Ensemble member filter
            if let Some(ens) = criteria.ensemble {
                if e.ensemble_member != Some(ens) {
                    return false;
                }
            }

            true
        })
        .collect()
}

/// Compute byte ranges for downloading specific entries from a GRIB2 file.
///
/// Each entry's data spans from its byte_offset to the next entry's byte_offset - 1.
/// The last selected entry extends to the end of the file (represented as u64::MAX).
///
/// The `entries` slice must be the full sorted list of idx entries so that the
/// "next entry" byte offset can be determined.
pub fn byte_ranges(entries: &[IdxEntry], selected: &[&IdxEntry]) -> Vec<(u64, u64)> {
    let mut ranges = Vec::with_capacity(selected.len());

    for sel in selected {
        let start = sel.byte_offset;

        // Find the next entry by looking for the entry with the next message number
        let end = entries
            .iter()
            .find(|e| e.byte_offset > start)
            .map(|e| e.byte_offset - 1)
            .unwrap_or(u64::MAX);

        ranges.push((start, end));
    }

    ranges
}

/// Discover which forecast hours are available for a model run by probing .idx files.
///
/// Sends HEAD requests to check for the existence of .idx files at each candidate
/// forecast hour. Uses parallel probing for speed.
///
/// # Arguments
///
/// * `client` - HTTP client for making HEAD requests
/// * `model` - Model name ("hrrr", "gfs", "nam", "rap")
/// * `date` - Date string in YYYYMMDD format
/// * `hour` - Model initialization hour (0-23)
/// * `product` - Product type ("sfc", "prs", "nat", "subh")
///
/// # Returns
///
/// Sorted vector of available forecast hours.
#[cfg(feature = "network")]
pub fn available_fhours(
    client: &super::DownloadClient,
    model: &str,
    date: &str,
    hour: u32,
    product: &str,
) -> Vec<u32> {
    use rayon::prelude::*;

    // Determine candidate forecast hours based on model
    let candidates: Vec<u32> = match model.to_lowercase().as_str() {
        "hrrr" => {
            // HRRR: 0-18 for most runs, 0-48 for 00/06/12/18z
            if hour % 6 == 0 {
                (0..=48).collect()
            } else {
                (0..=18).collect()
            }
        }
        "gfs" => {
            // GFS: 0-120 hourly, then 120-384 every 3 hours
            let mut hours: Vec<u32> = (0..=120).collect();
            hours.extend((123..=384).step_by(3));
            hours
        }
        "nam" => (0..=84).collect(),
        "rap" => (0..=21).collect(),
        "rrfs" => (0..=60).collect(),
        "nbm" => {
            let mut hours: Vec<u32> = (1..=36).collect();
            hours.extend((36..=264).step_by(3));
            hours
        }
        _ => (0..=48).collect(),
    };

    let idx_urls: Vec<(u32, String)> = candidates
        .iter()
        .map(|&fh| {
            let url = build_idx_url(model, date, hour, product, fh);
            (fh, url)
        })
        .collect();

    // Parallel HEAD requests to probe availability
    let mut available: Vec<u32> = idx_urls
        .par_iter()
        .filter(|(_, url)| client.head_ok(url))
        .map(|(fh, _)| *fh)
        .collect();

    available.sort();
    available
}

/// Build an .idx URL for a given model, date, hour, product, and forecast hour.
fn build_idx_url(model: &str, date: &str, hour: u32, product: &str, fhour: u32) -> String {
    use crate::models;

    match model.to_lowercase().as_str() {
        "hrrr" => models::HrrrConfig::idx_url(date, hour, product, fhour),
        "gfs" => models::GfsConfig::idx_url(date, hour, fhour),
        "nam" => models::NamConfig::idx_url(date, hour, fhour),
        "rap" => models::RapConfig::idx_url(date, hour, fhour),
        _ => {
            // Fallback: try HRRR-style
            models::HrrrConfig::idx_url(date, hour, product, fhour)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_IDX: &str = "\
1:0:d=2026031012:TMP:2 m above ground:anl:
2:47843:d=2026031012:TMP:surface:anl:
3:96542:d=2026031012:SPFH:2 m above ground:anl:
4:143210:d=2026031012:CAPE:surface:anl:
5:200000:d=2026031012:REFC:entire atmosphere:anl:
";

    const EXTENDED_IDX: &str = "\
101:234567:d=2026031012:APCP:surface:0-3 hr acc fcst:
102:300000:d=2026031012:TMP:500 mb:6 hr fcst:ENS=+5:
103:400000:d=2026031012:UGRD:10 m above ground:1 hr fcst:
104:500000:d=2026031012:TMAX:2 m above ground:0-6 hr max fcst:
105:600000:d=2026031012:PRATE:surface:0-1 hr avg fcst:
106:700000:d=2026031012:TMIN:2 m above ground:0-6 hr min fcst:
";

    #[test]
    fn test_parse_idx() {
        let entries = parse_idx(SAMPLE_IDX);
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[0].msg_num, 1);
        assert_eq!(entries[0].byte_offset, 0);
        assert_eq!(entries[0].date, "2026031012");
        assert_eq!(entries[0].variable, "TMP");
        assert_eq!(entries[0].level, "2 m above ground");
        assert_eq!(entries[0].forecast, "anl");
        assert!(entries[0].ensemble_member.is_none());
        assert!(entries[0].statistical_process.is_none());
        assert!(entries[0].time_range.is_none());

        assert_eq!(entries[1].byte_offset, 47843);
        assert_eq!(entries[1].variable, "TMP");
        assert_eq!(entries[1].level, "surface");
    }

    #[test]
    fn test_parse_extended_idx() {
        let entries = parse_idx(EXTENDED_IDX);
        assert_eq!(entries.len(), 6);

        // APCP accumulated
        assert_eq!(entries[0].variable, "APCP");
        assert_eq!(entries[0].statistical_process, Some("acc".to_string()));
        assert_eq!(entries[0].time_range, Some((0, 3)));

        // TMP with ensemble member
        assert_eq!(entries[1].variable, "TMP");
        assert_eq!(entries[1].ensemble_member, Some(5));
        assert!(entries[1].statistical_process.is_none());

        // TMAX
        assert_eq!(entries[3].statistical_process, Some("max".to_string()));
        assert_eq!(entries[3].time_range, Some((0, 6)));

        // PRATE avg
        assert_eq!(entries[4].statistical_process, Some("avg".to_string()));
        assert_eq!(entries[4].time_range, Some((0, 1)));

        // TMIN
        assert_eq!(entries[5].statistical_process, Some("min".to_string()));
    }

    #[test]
    fn test_find_entries() {
        let entries = parse_idx(SAMPLE_IDX);
        let found = find_entries(&entries, "TMP:2 m above ground");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].msg_num, 1);

        let found = find_entries(&entries, "TMP:surface");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].msg_num, 2);

        // Match all TMP entries
        let found = find_entries(&entries, "TMP");
        assert_eq!(found.len(), 2);

        let found = find_entries(&entries, "CAPE:surface");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_find_entries_regex() {
        let entries = parse_idx(SAMPLE_IDX);

        // Wildcard matching
        let found = find_entries_regex(&entries, "TMP.*surface");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].msg_num, 2);

        // Alternation
        let found = find_entries_regex(&entries, "(TMP|CAPE)");
        assert_eq!(found.len(), 3); // 2 TMP + 1 CAPE

        // Case-insensitive
        let found = find_entries_regex(&entries, "(?i)tmp.*surface");
        assert_eq!(found.len(), 1);

        // Alternation with suffix
        let found = find_entries_regex(&entries, "(TMP|SPFH).*2 m");
        assert_eq!(found.len(), 2); // TMP at 2m + SPFH at 2m
    }

    #[test]
    fn test_find_entries_criteria() {
        let entries = parse_idx(EXTENDED_IDX);

        // Filter by variable
        let criteria = SearchCriteria {
            variable: Some("APCP".to_string()),
            ..Default::default()
        };
        let found = find_entries_criteria(&entries, &criteria);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].variable, "APCP");

        // Filter by statistical process
        let criteria = SearchCriteria {
            statistical: Some("acc".to_string()),
            ..Default::default()
        };
        let found = find_entries_criteria(&entries, &criteria);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].variable, "APCP");

        // Filter by level type
        let criteria = SearchCriteria {
            level_type: Some("surface".to_string()),
            ..Default::default()
        };
        let found = find_entries_criteria(&entries, &criteria);
        assert_eq!(found.len(), 2); // APCP:surface + PRATE:surface

        // Filter by ensemble member
        let criteria = SearchCriteria {
            ensemble: Some(5),
            ..Default::default()
        };
        let found = find_entries_criteria(&entries, &criteria);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].variable, "TMP");
        assert_eq!(found[0].level, "500 mb");

        // Combined criteria
        let criteria = SearchCriteria {
            level_type: Some("pressure".to_string()),
            ensemble: Some(5),
            ..Default::default()
        };
        let found = find_entries_criteria(&entries, &criteria);
        assert_eq!(found.len(), 1);

        // Filter instantaneous (no statistical process)
        let criteria = SearchCriteria {
            statistical: Some("inst".to_string()),
            ..Default::default()
        };
        let found = find_entries_criteria(&entries, &criteria);
        assert_eq!(found.len(), 2); // TMP:500mb and UGRD:10m
    }

    #[test]
    fn test_byte_ranges() {
        let entries = parse_idx(SAMPLE_IDX);
        let selected = find_entries(&entries, "TMP:2 m above ground");
        let ranges = byte_ranges(&entries, &selected);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (0, 47842));

        let selected = find_entries(&entries, "REFC");
        let ranges = byte_ranges(&entries, &selected);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].0, 200000);
        assert_eq!(ranges[0].1, u64::MAX); // last entry
    }
}
