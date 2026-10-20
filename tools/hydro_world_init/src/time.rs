//! UTC ISO-8601 <-> seconds since 1970 (proleptic Gregorian, no leap seconds).
use crate::Result;

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// "2023-01-01T00:00:00Z" -> seconds since 1970-01-01T00:00:00Z.
pub fn parse(s: &str) -> Result<i64> {
    let b = s.as_bytes();
    if b.len() != 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' || b[19] != b'Z' {
        return Err(format!("time {s:?} is not YYYY-MM-DDTHH:MM:SSZ").into());
    }
    let n = |a: usize, z: usize| -> Result<i64> { Ok(s[a..z].parse::<i64>()?) };
    let (y, mo, d, h, mi, se) = (n(0, 4)?, n(5, 7)?, n(8, 10)?, n(11, 13)?, n(14, 16)?, n(17, 19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || se > 59 {
        return Err(format!("time {s:?} out of range").into());
    }
    Ok(days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + se)
}

pub fn iso(t: i64) -> String {
    let (y, m, d) = civil_from_days(t.div_euclid(86_400));
    let s = t.rem_euclid(86_400);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", s / 3600, (s / 60) % 60, s % 60)
}

#[cfg(test)]
mod tests {
    #[test]
    fn roundtrip() {
        for s in ["1970-01-01T00:00:00Z", "2023-01-01T00:00:00Z", "2022-12-02T01:00:00Z", "2000-02-29T23:59:59Z"] {
            assert_eq!(super::iso(super::parse(s).unwrap()), s);
        }
        assert_eq!(super::parse("2023-01-01T00:00:00Z").unwrap(), 1_672_531_200);
    }
}
