use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoesSatellite {
    G16,
    G17,
    G18,
    G19,
    Other(String),
}

impl GoesSatellite {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_uppercase().as_str() {
            "G16" | "GOES16" | "GOES-16" => Self::G16,
            "G17" | "GOES17" | "GOES-17" => Self::G17,
            "G18" | "GOES18" | "GOES-18" => Self::G18,
            "G19" | "GOES19" | "GOES-19" => Self::G19,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::G16 => "G16",
            Self::G17 => "G17",
            Self::G18 => "G18",
            Self::G19 => "G19",
            Self::Other(value) => value.as_str(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoesAbiFilename {
    pub product: String,
    pub mode: Option<u8>,
    pub channel: Option<u8>,
    pub satellite: GoesSatellite,
    pub start_time_utc: DateTime<Utc>,
    pub end_time_utc: DateTime<Utc>,
    pub created_time_utc: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoesParseError(String);

impl fmt::Display for GoesParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GoesParseError {}

pub fn parse_goes_abi_filename(path: impl AsRef<Path>) -> Result<GoesAbiFilename, GoesParseError> {
    let name = path
        .as_ref()
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| GoesParseError("GOES filename is not valid UTF-8".to_string()))?;
    let name = name.strip_suffix(".nc").unwrap_or(name);
    let mut parts = name.split('_');
    match parts.next() {
        Some("OR") => {}
        _ => {
            return Err(GoesParseError(format!(
                "GOES filename missing OR product prefix: {name}"
            )));
        }
    }
    let product_part = parts
        .next()
        .ok_or_else(|| GoesParseError(format!("GOES filename missing product id: {name}")))?;
    let satellite = parts
        .next()
        .map(GoesSatellite::parse)
        .ok_or_else(|| GoesParseError(format!("GOES filename missing satellite id: {name}")))?;
    let start = parts
        .next()
        .ok_or_else(|| GoesParseError(format!("GOES filename missing start time: {name}")))?;
    let end = parts
        .next()
        .ok_or_else(|| GoesParseError(format!("GOES filename missing end time: {name}")))?;
    let created = parts
        .next()
        .ok_or_else(|| GoesParseError(format!("GOES filename missing created time: {name}")))?;

    let (product, mode, channel) = parse_product_mode_channel(product_part);
    Ok(GoesAbiFilename {
        product,
        mode,
        channel,
        satellite,
        start_time_utc: parse_goes_timestamp(start)?,
        end_time_utc: parse_goes_timestamp(end)?,
        created_time_utc: parse_goes_timestamp(created)?,
    })
}

fn parse_product_mode_channel(value: &str) -> (String, Option<u8>, Option<u8>) {
    let Some((product, mode_part)) = value.rsplit_once('-') else {
        return (value.to_string(), None, None);
    };
    let Some(rest) = mode_part.strip_prefix('M') else {
        return (value.to_string(), None, None);
    };
    let (mode_raw, channel_raw) = match rest.split_once('C') {
        Some((mode, channel)) => (mode, Some(channel)),
        None => (rest, None),
    };
    let mode = mode_raw.parse::<u8>().ok();
    let channel = channel_raw.and_then(|raw| raw.parse::<u8>().ok());
    (product.to_string(), mode, channel)
}

fn parse_goes_timestamp(value: &str) -> Result<DateTime<Utc>, GoesParseError> {
    let raw = value
        .strip_prefix(['s', 'e', 'c'])
        .ok_or_else(|| GoesParseError(format!("GOES timestamp missing prefix: {value}")))?;
    if raw.len() < 14 {
        return Err(GoesParseError(format!("GOES timestamp too short: {value}")));
    }
    let year = parse_i32(&raw[0..4], value)?;
    let doy = parse_u32(&raw[4..7], value)?;
    let hour = parse_u32(&raw[7..9], value)?;
    let minute = parse_u32(&raw[9..11], value)?;
    let second = parse_u32(&raw[11..13], value)?;
    let tenth = parse_u32(&raw[13..14], value)?;
    let date = NaiveDate::from_yo_opt(year, doy)
        .ok_or_else(|| GoesParseError(format!("invalid GOES day-of-year timestamp: {value}")))?;
    let naive = date
        .and_hms_milli_opt(hour, minute, second, tenth * 100)
        .ok_or_else(|| GoesParseError(format!("invalid GOES timestamp clock: {value}")))?;
    Ok(Utc.from_utc_datetime(&naive))
}

fn parse_i32(raw: &str, source: &str) -> Result<i32, GoesParseError> {
    raw.parse::<i32>()
        .map_err(|_| GoesParseError(format!("invalid GOES timestamp component in {source}")))
}

fn parse_u32(raw: &str, source: &str) -> Result<u32, GoesParseError> {
    raw.parse::<u32>()
        .map_err(|_| GoesParseError(format!("invalid GOES timestamp component in {source}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_goes_abi_filename() {
        let parsed = parse_goes_abi_filename(
            "OR_ABI-L2-CMIPC-M6C13_G18_s20261180646171_e20261180648556_c20261180649033.nc",
        )
        .unwrap();
        assert_eq!(parsed.product, "ABI-L2-CMIPC");
        assert_eq!(parsed.mode, Some(6));
        assert_eq!(parsed.channel, Some(13));
        assert_eq!(parsed.satellite, GoesSatellite::G18);
        assert_eq!(
            parsed.start_time_utc.to_rfc3339(),
            "2026-04-28T06:46:17.100+00:00"
        );
    }

    #[test]
    fn parses_multiband_goes_abi_filename() {
        let parsed = parse_goes_abi_filename(
            "OR_ABI-L2-MCMIPC-M6_G18_s20261180601171_e20261180603544_c20261180604252.nc",
        )
        .unwrap();
        assert_eq!(parsed.product, "ABI-L2-MCMIPC");
        assert_eq!(parsed.mode, Some(6));
        assert_eq!(parsed.channel, None);
    }
}
