/// Detect the latest available model run by probing NOMADS .idx files.
///
/// Strategy: start from the current UTC hour and work backwards up to 48 hours,
/// checking whether the f00 .idx file exists on NOMADS (NCEP's operational server).
/// NOMADS is the authoritative source — new runs appear there first.

use chrono::{Utc, TimeDelta};
use crate::download::DownloadClient;
use crate::models::{HrrrConfig, GfsConfig, NamConfig, RapConfig};

/// Maximum number of hours to search backwards.
const MAX_LOOKBACK_HOURS: i64 = 48;

/// Find the most recent available run for a given model.
///
/// Returns `(date_str, hour)` where `date_str` is `"YYYYMMDD"` and `hour` is
/// the initialization hour (0-23). Probes for the f00 .idx file on NOMADS.
///
/// # Errors
///
/// Returns an error if no run is found within the last 48 hours, or if the
/// model name is not recognized.
pub fn find_latest_run(client: &DownloadClient, model: &str) -> Result<(String, u32), String> {
    let model_lower = model.to_lowercase();

    // Determine valid init hours for this model
    let valid_hours: &[u32] = match model_lower.as_str() {
        "hrrr" | "rap" => &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        ],
        "gfs" | "nam" => &[0, 6, 12, 18],
        _ => {
            return Err(format!(
                "Unknown model '{}'. Supported for latest-run detection: hrrr, gfs, nam, rap",
                model
            ));
        }
    };

    let now = Utc::now();
    let mut checked = 0u32;

    // Walk backwards hour by hour, but only probe valid init hours
    for lookback in 0..MAX_LOOKBACK_HOURS {
        let candidate = now - TimeDelta::hours(lookback);
        let hour = candidate.format("%H").to_string().parse::<u32>().unwrap();

        if !valid_hours.contains(&hour) {
            continue;
        }

        let date_str = candidate.format("%Y%m%d").to_string();

        // Probe NOMADS — it's the authoritative source and gets new runs first
        let idx_url = match model_lower.as_str() {
            "hrrr" => format!("{}.idx", HrrrConfig::nomads_url(&date_str, hour, "sfc", 0)),
            "gfs" => format!("{}.idx", GfsConfig::nomads_url(&date_str, hour, 0)),
            "nam" => format!("{}.idx", NamConfig::nomads_url(&date_str, hour, 0)),
            "rap" => format!("{}.idx", RapConfig::nomads_url(&date_str, hour, 0)),
            _ => unreachable!(),
        };

        checked += 1;
        eprintln!("  Probing {} {:02}z ...", date_str, hour);

        if client.head_ok(&idx_url) {
            eprintln!("  Found latest {} run: {}/{:02}z", model, date_str, hour);
            return Ok((date_str, hour));
        }
    }

    Err(format!(
        "No {} run found in the last {} hours ({} candidates checked)",
        model, MAX_LOOKBACK_HOURS, checked
    ))
}

#[cfg(test)]
mod tests {
    use chrono::TimeDelta;

    #[test]
    fn test_timedelta_subtraction() {
        // Verify chrono TimeDelta works as expected for hour subtraction
        let now = chrono::NaiveDate::from_ymd_opt(2026, 3, 10)
            .unwrap()
            .and_hms_opt(2, 0, 0)
            .unwrap();
        let earlier = now - TimeDelta::hours(5);
        assert_eq!(earlier.format("%Y%m%d").to_string(), "20260309");
        assert_eq!(earlier.format("%H").to_string(), "21");
    }
}
