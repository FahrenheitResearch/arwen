use crate::{
    finite,
    score::{Period, Row},
    Result,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputFile {
    pub path: String,
    pub day: String,
    pub domain: String,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default)]
    pub start_valid_exclusive: Option<String>,
    #[serde(default)]
    pub end_valid_inclusive: Option<String>,
}

fn default_method() -> String {
    "bilinear".into()
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportRequest {
    pub files: Vec<InputFile>,
    pub expected_members: Vec<String>,
    pub timezone_offset_hours: i32,
}

#[derive(Clone, Debug, Deserialize)]
struct CsvRow {
    station: String,
    valid_time: String,
    arm: String,
    method: String,
    observed_k: String,
    forecast_k: String,
    status: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct FileAudit {
    pub path: String,
    pub day: String,
    pub domain: String,
    pub method: String,
    pub start_valid_exclusive: String,
    pub end_valid_inclusive: String,
    pub total_rows: usize,
    pub paired_rows: usize,
    pub usable_rows: usize,
    pub rejected_status_rows: usize,
    pub rejected_method_rows: usize,
    pub excluded_member_rows: usize,
    pub outside_window_rows: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct DomainSelection {
    pub day: String,
    pub station: String,
    pub valid: String,
    pub domain: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImportResult {
    pub rows: Vec<Row>,
    pub files: Vec<FileAudit>,
    pub keys_considered: usize,
    pub keys_selected: usize,
    pub keys_without_common_domain: usize,
    pub selected_domains: BTreeMap<String, usize>,
    pub domain_support: Vec<DomainSelection>,
    pub units: &'static str,
    pub period_rule: &'static str,
    pub domain_rule: &'static str,
}

#[derive(Clone, Debug)]
struct Timestamp {
    seconds: i64,
    hour: i32,
    canonical: String,
}

fn date_ordinal(date: &str) -> Result<i64> {
    if !date.is_ascii() || date.len() != 10 || &date[4..5] != "-" || &date[7..8] != "-" {
        return Err(format!("date must be YYYY-MM-DD: {date}"));
    }
    let number = |start, end| {
        date[start..end]
            .parse::<i64>()
            .map_err(|_| format!("invalid date: {date}"))
    };
    let year = number(0, 4)?;
    let month = number(5, 7)?;
    let day = number(8, 10)?;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if year < 1 || !(1..=12).contains(&month) || day < 1 || day > month_days[(month - 1) as usize] {
        return Err(format!("invalid calendar date: {date}"));
    }
    let previous_year = year - 1;
    let mut ordinal =
        365 * previous_year + previous_year / 4 - previous_year / 100 + previous_year / 400;
    ordinal += month_days[..(month - 1) as usize].iter().sum::<i64>() + day - 1;
    Ok(ordinal)
}

fn parse_timestamp(value: &str) -> Result<Timestamp> {
    if !value.is_ascii()
        || value.len() < 19
        || !matches!(value.as_bytes()[10], b'T' | b' ')
        || &value[13..14] != ":"
        || &value[16..17] != ":"
    {
        return Err(format!("UTC timestamp needs YYYY-MM-DDTHH:MM:SS: {value}"));
    }
    let date = &value[..10];
    let ordinal = date_ordinal(date)?;
    let number = |start, end| {
        value[start..end]
            .parse::<i32>()
            .map_err(|_| format!("invalid time: {value}"))
    };
    let hour = number(11, 13)?;
    let minute = number(14, 16)?;
    let second = number(17, 19)?;
    if !(0..24).contains(&hour) || !(0..60).contains(&minute) || !(0..60).contains(&second) {
        return Err(format!("invalid time: {value}"));
    }
    let suffix = &value[19..];
    if suffix != "" && suffix != "Z" && suffix != "+00:00" {
        return Err(format!("timestamp must use whole UTC seconds: {value}"));
    }
    Ok(Timestamp {
        seconds: ordinal * 86_400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64,
        hour,
        canonical: format!("{date}T{hour:02}:{minute:02}:{second:02}Z"),
    })
}

fn next_date(date: &str) -> Result<String> {
    date_ordinal(date)?;
    let year: i32 = date[..4].parse().map_err(|_| "invalid year")?;
    let month: i32 = date[5..7].parse().map_err(|_| "invalid month")?;
    let day: i32 = date[8..10].parse().map_err(|_| "invalid day")?;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let lengths = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let (year, month, day) = if day < lengths[(month - 1) as usize] {
        (year, month, day + 1)
    } else if month < 12 {
        (year, month + 1, 1)
    } else {
        (year + 1, 1, 1)
    };
    if year > 9999 {
        return Err("date window exceeds four-digit years".into());
    }
    Ok(format!("{year:04}-{month:02}-{day:02}"))
}

fn domain_rank(domain: &str) -> Result<usize> {
    let suffix = domain
        .strip_prefix('d')
        .ok_or_else(|| format!("domain needs dNN prefix: {domain}"))?;
    let digits: String = suffix.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return Err(format!("domain needs dNN prefix: {domain}"));
    }
    let rank = digits
        .parse::<usize>()
        .map_err(|_| format!("invalid domain: {domain}"))?;
    if rank == 0 {
        return Err("domain rank must be positive".into());
    }
    Ok(rank)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    day: String,
    station: String,
    valid: String,
}

struct Prepared {
    expected: BTreeSet<String>,
    rows: BTreeMap<(Key, usize, String), Row>,
    domains: BTreeMap<usize, String>,
    files: Vec<FileAudit>,
    timezone: i32,
}

impl Prepared {
    fn add_file(&mut self, file: InputFile, reader: impl Read) -> Result<()> {
        date_ordinal(&file.day)?;
        if file.method.trim().is_empty() {
            return Err("sampling method must be nonempty".into());
        }
        let rank = domain_rank(&file.domain)?;
        if self
            .domains
            .get(&rank)
            .is_some_and(|label| *label != file.domain)
        {
            return Err(format!("two domain labels share rank {rank}"));
        }
        self.domains.insert(rank, file.domain.clone());
        let start = parse_timestamp(
            &file
                .start_valid_exclusive
                .unwrap_or_else(|| format!("{}T00:00:00Z", file.day)),
        )?;
        let end = parse_timestamp(
            &file
                .end_valid_inclusive
                .unwrap_or(format!("{}T00:00:00Z", next_date(&file.day)?)),
        )?;
        if start.seconds >= end.seconds {
            return Err("valid-time window must increase".into());
        }
        let mut audit = FileAudit {
            path: file.path,
            day: file.day.clone(),
            domain: file.domain,
            method: file.method.clone(),
            start_valid_exclusive: start.canonical.clone(),
            end_valid_inclusive: end.canonical.clone(),
            ..FileAudit::default()
        };
        let mut csv = csv::ReaderBuilder::new().from_reader(reader);
        let header = csv
            .headers()
            .map_err(|e| format!("invalid CSV header: {e}"))?;
        for required in [
            "station",
            "valid_time",
            "arm",
            "method",
            "observed_k",
            "forecast_k",
            "status",
        ] {
            if !header.iter().any(|column| column == required) {
                return Err(format!("CSV missing {required} column"));
            }
        }
        for record in csv.deserialize::<CsvRow>() {
            let record = record.map_err(|e| format!("invalid station CSV row: {e}"))?;
            audit.total_rows += 1;
            if record.status != "paired" {
                audit.rejected_status_rows += 1;
                continue;
            }
            audit.paired_rows += 1;
            if record.method != file.method {
                audit.rejected_method_rows += 1;
                continue;
            }
            if !self.expected.contains(&record.arm) {
                audit.excluded_member_rows += 1;
                continue;
            }
            let valid = parse_timestamp(&record.valid_time)?;
            if valid.seconds <= start.seconds || valid.seconds > end.seconds {
                audit.outside_window_rows += 1;
                continue;
            }
            if record.station.trim().is_empty() {
                return Err("CSV station must be nonempty".into());
            }
            let forecast = finite(
                record
                    .forecast_k
                    .parse::<f64>()
                    .map_err(|_| "paired row needs numeric forecast_k")?,
                "forecast_k",
            )?;
            let observation = finite(
                record
                    .observed_k
                    .parse::<f64>()
                    .map_err(|_| "paired row needs numeric observed_k")?,
                "observed_k",
            )?;
            let local_hour = (valid.hour + self.timezone).rem_euclid(24);
            let period = if (6..18).contains(&local_hour) {
                Period::Day
            } else {
                Period::Night
            };
            let key = Key {
                day: file.day.clone(),
                station: record.station,
                valid: valid.canonical,
            };
            let row = Row {
                member: record.arm.clone(),
                day: key.day.clone(),
                station: key.station.clone(),
                valid: key.valid.clone(),
                period,
                forecast,
                observation,
            };
            if self.rows.insert((key, rank, record.arm), row).is_some() {
                return Err("duplicate station/member/valid/domain row in input CSVs".into());
            }
            audit.usable_rows += 1;
        }
        self.files.push(audit);
        Ok(())
    }

    fn finish(self) -> Result<ImportResult> {
        let mut support: BTreeMap<Key, BTreeMap<usize, BTreeSet<String>>> = BTreeMap::new();
        let mut truth: BTreeMap<Key, f64> = BTreeMap::new();
        for ((key, domain, member), row) in &self.rows {
            if truth
                .get(key)
                .is_some_and(|observation| *observation != row.observation)
            {
                return Err(format!(
                    "observation mismatch between members or domains at {} {} {}",
                    key.day, key.station, key.valid
                ));
            }
            truth.insert(key.clone(), row.observation);
            support
                .entry(key.clone())
                .or_default()
                .entry(*domain)
                .or_default()
                .insert(member.clone());
        }
        let mut rows = Vec::new();
        let mut domain_support = Vec::new();
        let mut selected_domains = BTreeMap::new();
        for (key, domains) in &support {
            let common = domains
                .iter()
                .rev()
                .find(|(_, members)| **members == self.expected);
            if let Some((&rank, _)) = common {
                let domain = self.domains[&rank].clone();
                *selected_domains.entry(domain.clone()).or_insert(0) += 1;
                domain_support.push(DomainSelection {
                    day: key.day.clone(),
                    station: key.station.clone(),
                    valid: key.valid.clone(),
                    domain,
                });
                for member in &self.expected {
                    rows.push(self.rows[&(key.clone(), rank, member.clone())].clone());
                }
            }
        }
        let keys_considered = support.len();
        let keys_selected = domain_support.len();
        Ok(ImportResult { rows, files: self.files, keys_considered, keys_selected,
            keys_without_common_domain: keys_considered - keys_selected, selected_domains, domain_support,
            units: "K", period_rule: "day is local hour 06 inclusive to 18 exclusive; otherwise night",
            domain_rule: "highest numeric dNN domain with paired requested-method rows for every expected member" })
    }
}

pub fn import(request: ImportRequest) -> Result<ImportResult> {
    if request.files.is_empty() || request.expected_members.is_empty() {
        return Err("import needs files and expected members".into());
    }
    if !(-23..=23).contains(&request.timezone_offset_hours) {
        return Err("timezone offset must be between -23 and 23 hours".into());
    }
    let mut expected = BTreeSet::new();
    for member in request.expected_members {
        if member.trim().is_empty() || !expected.insert(member) {
            return Err("expected members must be nonempty and unique".into());
        }
    }
    let mut prepared = Prepared {
        expected,
        rows: BTreeMap::new(),
        domains: BTreeMap::new(),
        files: Vec::new(),
        timezone: request.timezone_offset_hours,
    };
    for file in request.files {
        let reader = std::fs::File::open(&file.path)
            .map_err(|e| format!("cannot open station CSV {}: {e}", file.path))?;
        prepared.add_file(file, reader)?;
    }
    prepared.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared() -> Prepared {
        Prepared {
            expected: ["a".into(), "b".into()].into_iter().collect(),
            rows: BTreeMap::new(),
            domains: BTreeMap::new(),
            files: Vec::new(),
            timezone: -7,
        }
    }

    fn file(domain: &str) -> InputFile {
        InputFile {
            path: "hourly.csv".into(),
            day: "2024-02-28".into(),
            domain: domain.into(),
            method: "bilinear".into(),
            start_valid_exclusive: None,
            end_valid_inclusive: None,
        }
    }

    const HEADER: &str = "station,valid_time,arm,method,observed_k,forecast_k,status\n";

    #[test]
    fn selects_finest_common_domain_and_classifies_local_period() {
        let coarse = format!("{HEADER}x,2024-02-28T14:00:00,a,bilinear,280,281,paired\nx,2024-02-28T14:00:00,b,bilinear,280,282,paired\ny,2024-02-28T00:00:00,a,bilinear,280,281,paired\ny,2024-02-29T00:00:00,a,bilinear,280,283,paired\ny,2024-02-29T00:00:00,b,bilinear,280,284,paired\n");
        let fine = format!("{HEADER}x,2024-02-28T14:00:00,a,bilinear,280,285,paired\nx,2024-02-28T14:00:00,b,bilinear,280,286,paired\ny,2024-02-29T00:00:00,a,bilinear,280,287,paired\nz,2024-02-28T18:00:00,a,bilinear,,,missing\n");
        let mut prepared = prepared();
        prepared.add_file(file("d01"), coarse.as_bytes()).unwrap();
        prepared.add_file(file("d02"), fine.as_bytes()).unwrap();
        let result = prepared.finish().unwrap();
        assert_eq!(result.keys_selected, 2);
        assert_eq!(result.rows.len(), 4);
        assert_eq!(result.selected_domains["d02"], 1);
        assert_eq!(result.selected_domains["d01"], 1);
        assert_eq!(result.rows[0].forecast, 285.0);
        assert_eq!(result.rows[0].period, Period::Day);
        assert_eq!(result.files[0].outside_window_rows, 1);
        assert_eq!(result.files[1].rejected_status_rows, 1);
    }

    #[test]
    fn missing_common_domain_is_counted_and_duplicates_rejected() {
        let csv = format!("{HEADER}x,2024-02-28T03:00:00,a,bilinear,280,281,paired\n");
        let mut input = prepared();
        input.add_file(file("d01"), csv.as_bytes()).unwrap();
        let result = input.finish().unwrap();
        assert_eq!(result.keys_without_common_domain, 1);
        assert!(result.rows.is_empty());
        let mut input = prepared();
        input.add_file(file("d01"), csv.as_bytes()).unwrap();
        assert!(input
            .add_file(file("d01"), csv.as_bytes())
            .unwrap_err()
            .contains("duplicate"));
    }

    #[test]
    fn timestamps_require_real_utc_calendar_dates() {
        assert!(parse_timestamp("2023-02-29T12:00:00").is_err());
        assert!(parse_timestamp("2024-02-28T25:00:00").is_err());
        assert!(parse_timestamp("2024-02-28T12:00:00-07:00").is_err());
        assert_eq!(next_date("2024-02-28").unwrap(), "2024-02-29");
        assert_eq!(next_date("2024-02-29").unwrap(), "2024-03-01");
        assert_eq!(next_date("2024-12-31").unwrap(), "2025-01-01");
    }

    #[test]
    fn mismatched_observation_across_domains_is_rejected() {
        let csv1 = format!("{HEADER}x,2024-02-28T03:00:00,a,bilinear,280,281,paired\n");
        let csv2 = format!("{HEADER}x,2024-02-28T03:00:00,b,bilinear,281,282,paired\n");
        let mut input = prepared();
        input.add_file(file("d01"), csv1.as_bytes()).unwrap();
        input.add_file(file("d02"), csv2.as_bytes()).unwrap();
        assert!(input.finish().unwrap_err().contains("observation mismatch"));
    }
}
