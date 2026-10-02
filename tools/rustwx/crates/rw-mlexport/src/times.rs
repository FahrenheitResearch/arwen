//! UTC instants as whole seconds since 1970, and the spellings history files
//! and requests use for them.  Proleptic Gregorian, no leap seconds: the
//! calendar CF calls `proleptic_gregorian` and WRF's history files use.

use crate::error::{refuse, Result};

/// Days from 1970-01-01 to `y-m-d` (Howard Hinnant's `days_from_civil`).
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `(y, m, d)` for days since 1970-01-01 (Hinnant's `civil_from_days`).
pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn digits(s: &[u8]) -> Option<i64> {
    if s.is_empty() || !s.iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(s).ok()?.parse().ok()
}

/// Parse `YYYY-MM-DD?HH:MM:SS` where `?` is `_`, `T` or a space and the time
/// separators are `:` or `_` (a history file name on a file system that
/// refuses colons).  A trailing `Z` is accepted.  None for anything else.
pub fn parse(text: &str) -> Option<i64> {
    let t = text.trim().trim_end_matches('Z');
    let b = t.as_bytes();
    if b.len() != 19 {
        return None;
    }
    if b[4] != b'-' || b[7] != b'-' || !matches!(b[10], b'_' | b'T' | b' ') {
        return None;
    }
    if !matches!(b[13], b':' | b'_') || !matches!(b[16], b':' | b'_') {
        return None;
    }
    let (y, mo, d) = (digits(&b[0..4])?, digits(&b[5..7])?, digits(&b[8..10])?);
    let (h, mi, s) = (digits(&b[11..13])?, digits(&b[14..16])?, digits(&b[17..19])?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 60 {
        return None;
    }
    let days = days_from_civil(y, mo, d);
    let (cy, cm, cd) = civil_from_days(days);
    if (cy, cm, cd) != (y, mo, d) {
        return None; // 2026-02-30 and friends
    }
    Some(days * 86_400 + h * 3600 + mi * 60 + s)
}

/// Like [`parse`], refusing with a sentence naming `what` on failure.
pub fn parse_named(text: &str, what: &str) -> Result<i64> {
    parse(text).ok_or_else(|| {
        refuse(format!(
            "{what} '{text}' is not a YYYY-MM-DD_HH:MM:SS time, so the frame could not be placed on the time axis"
        ))
    })
}

fn split(seconds: i64) -> (i64, i64, i64, i64, i64, i64) {
    let days = seconds.div_euclid(86_400);
    let rem = seconds.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    (y, m, d, rem / 3600, (rem % 3600) / 60, rem % 60)
}

/// `YYYY-MM-DDTHH:MM:SS` (ISO 8601, the spelling CF time units take).
pub fn iso(seconds: i64) -> String {
    let (y, m, d, h, mi, s) = split(seconds);
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}")
}

/// `YYYY-MM-DD HH:MM:SS` (the spelling CF `units` strings use after `since`).
pub fn cf(seconds: i64) -> String {
    let (y, m, d, h, mi, s) = split(seconds);
    format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}:{s:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_spelling() {
        let t = parse("2026-09-29_08:00:00").unwrap();
        assert_eq!(parse("2026-09-29T08:00:00Z"), Some(t));
        assert_eq!(parse("2026-09-29 08:00:00"), Some(t));
        assert_eq!(parse("2026-09-29_08_00_00"), Some(t));
        assert_eq!(iso(t), "2026-09-29T08:00:00");
        assert_eq!(cf(t), "2026-09-29 08:00:00");
        assert_eq!(parse("1970-01-01_00:00:00"), Some(0));
        assert_eq!(parse("2000-03-01_00:00:00"), Some(951_868_800));
    }

    #[test]
    fn refuses_impossible_dates() {
        assert_eq!(parse("2026-02-30_00:00:00"), None);
        assert_eq!(parse("2026-13-01_00:00:00"), None);
        assert_eq!(parse("2026-09-29"), None);
        assert!(parse_named("garbage", "Times").is_err());
    }
}
