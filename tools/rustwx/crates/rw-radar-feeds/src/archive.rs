//! Dated archive lookup for the two feeds whose upstreams keep history that
//! BowEcho's live adapters do not reach.
//!
//! * **FMI** (Finland): the AWS Open Data bucket
//!   `fmi-opendata-radar-volume-hdf5` keeps every volume from 2007 under
//!   `{yyyy}/{mm}/{dd}/{site}/{yyyymmddhhmm}_{site}_PVOL.h5`, the same layout
//!   BowEcho's `FmiProvider` lists for today and yesterday (probed
//!   2026-10-16: year prefixes 2007..2026; `2010/06/01/fivan/` holds
//!   `201006010000_fivan_PVOL.h5`).
//! * **DMI** (Denmark): the radardata STAC items endpoint BowEcho's
//!   `DmiProvider` polls also answers `datetime=start/end` (probed
//!   2026-10-16: station 06177, 2026-05-01T00:00/00:30 returned the seven
//!   five-minute volumes). DMI keeps about five to six months there. It
//!   accepts only `sortorder=datetime,DESC` (ASC is HTTP 400), so the day is
//!   asked newest first and sorted here.
//!
//! Both wrap BowEcho's provider, so the live listing is BowEcho's unchanged.

use chrono::{Datelike, NaiveDate};
use data_source::international::{ArchiveFrames, FramePlan, IntlProvider, PlanPart};

use crate::delegate_provider;

const FMI_BUCKET: &str = "https://fmi-opendata-radar-volume-hdf5.s3.amazonaws.com";
const DMI_ITEMS: &str = "https://opendataapi.dmi.dk/v1/radardata/collections/volume/items";
/// One day at DMI's 5-minute cadence is 288 volumes; one page holds a day.
const DMI_DAY_LIMIT: usize = 1000;
/// A runaway pagination guard: one site-day of FMI keys fits one page.
const MAX_LISTING_PAGES: usize = 20;

/// FMI's provider with the bucket's 2007-onward archive.
pub struct FmiArchive {
    inner: Box<dyn IntlProvider>,
}

impl FmiArchive {
    pub fn new(inner: Box<dyn IntlProvider>) -> Self {
        Self { inner }
    }
}

delegate_provider!(FmiArchive);

fn site_code_ok(site: &str) -> Result<(), String> {
    if !site.is_empty()
        && site
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    {
        Ok(())
    } else {
        Err(format!("invalid site code {site:?}"))
    }
}

fn percent_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// The text between `<tag>` and `</tag>`, every occurrence, in order.
fn xml_values<'a>(xml: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(at) = rest.find(&open) {
        let after = &rest[at + open.len()..];
        let Some(end) = after.find(&close) else { break };
        out.push(&after[..end]);
        rest = &after[end + close.len()..];
    }
    out
}

/// Every key under `prefix`, following continuation tokens.
fn s3_keys(prefix: &str) -> Result<Vec<String>, String> {
    let mut keys = Vec::new();
    let mut token: Option<String> = None;
    for _ in 0..MAX_LISTING_PAGES {
        let mut url = format!(
            "{FMI_BUCKET}/?list-type=2&max-keys=1000&prefix={}",
            percent_encode(prefix)
        );
        if let Some(t) = &token {
            url.push_str(&format!("&continuation-token={}", percent_encode(t)));
        }
        let xml = data_source::fetch_listing_text(&url).map_err(|e| format!("{url}: {e}"))?;
        keys.extend(xml_values(&xml, "Key").into_iter().map(str::to_owned));
        let truncated = xml_values(&xml, "IsTruncated").first().copied() == Some("true");
        token = xml_values(&xml, "NextContinuationToken")
            .first()
            .map(|t| t.to_string());
        if !truncated || token.is_none() {
            return Ok(keys);
        }
    }
    Err(format!(
        "listing {prefix} ran past {MAX_LISTING_PAGES} pages; refusing a partial day"
    ))
}

fn fmi_plans_from_keys(keys: &[String], site: &str) -> Vec<FramePlan> {
    let suffix = format!("_{site}_PVOL.h5");
    let mut keys: Vec<&String> = keys.iter().filter(|k| k.ends_with(&suffix)).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .map(|key| FramePlan {
            identity: key.rsplit('/').next().unwrap_or(key).to_owned(),
            parts: vec![PlanPart {
                url: format!("{FMI_BUCKET}/{key}"),
            }],
            merge: false,
        })
        .collect()
}

impl ArchiveFrames for FmiArchive {
    fn day_plans(&self, site_id: &str, date_utc: NaiveDate) -> Result<Vec<FramePlan>, String> {
        site_code_ok(site_id)?;
        let prefix = format!(
            "{:04}/{:02}/{:02}/{site_id}/",
            date_utc.year(),
            date_utc.month(),
            date_utc.day()
        );
        let keys = s3_keys(&prefix).map_err(|e| format!("FMI archive: {e}"))?;
        let plans = fmi_plans_from_keys(&keys, site_id);
        if plans.is_empty() {
            return Err(format!(
                "FMI archive holds no PVOL for {site_id} on {date_utc}"
            ));
        }
        Ok(plans)
    }
}

/// DMI's provider with the STAC's dated lookup.
pub struct DmiArchive {
    inner: Box<dyn IntlProvider>,
}

impl DmiArchive {
    pub fn new(inner: Box<dyn IntlProvider>) -> Self {
        Self { inner }
    }
}

delegate_provider!(DmiArchive);

fn dmi_plans_from_items(json: &str) -> Result<Vec<FramePlan>, String> {
    let doc: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("DMI items JSON: {e}"))?;
    let features = doc["features"]
        .as_array()
        .ok_or("DMI items answer has no features array")?;
    let mut plans = Vec::with_capacity(features.len());
    for feature in features {
        let id = feature["id"].as_str().unwrap_or_default();
        // DMI emits a singular "asset"; standard STAC spells it "assets".
        let href = feature["asset"]["data"]["href"]
            .as_str()
            .or_else(|| feature["assets"]["data"]["href"].as_str());
        let (false, Some(href)) = (id.is_empty(), href) else {
            return Err(format!("DMI item {id:?} has no id or no data href"));
        };
        plans.push(FramePlan {
            identity: id.to_owned(),
            parts: vec![PlanPart {
                url: href.to_owned(),
            }],
            merge: false,
        });
    }
    plans.sort_by(|a, b| a.identity.cmp(&b.identity));
    plans.dedup_by(|a, b| a.identity == b.identity);
    Ok(plans)
}

impl ArchiveFrames for DmiArchive {
    fn day_plans(&self, site_id: &str, date_utc: NaiveDate) -> Result<Vec<FramePlan>, String> {
        if site_id.is_empty() || !site_id.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!("DMI: invalid station id {site_id:?}"));
        }
        let day = date_utc.format("%Y-%m-%d");
        let url = format!(
            "{DMI_ITEMS}?stationId={site_id}&datetime={day}T00:00:00Z/{day}T23:59:59Z\
             &limit={DMI_DAY_LIMIT}&sortorder=datetime%2CDESC"
        );
        let json = data_source::fetch_text(&url).map_err(|e| format!("DMI archive {url}: {e}"))?;
        let plans = dmi_plans_from_items(&json)?;
        if plans.len() >= DMI_DAY_LIMIT {
            return Err(format!(
                "DMI answered {} items for {site_id} on {day}, the page limit; refusing a \
                 possibly partial day",
                plans.len()
            ));
        }
        if plans.is_empty() {
            return Err(format!(
                "DMI archive holds no volume for {site_id} on {day}"
            ));
        }
        Ok(plans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3_listing_keys_and_continuation_are_read() {
        let xml = "<ListBucketResult><IsTruncated>true</IsTruncated>\
            <Contents><Key>2010/06/01/fivan/201006010000_fivan_PVOL.h5</Key><Size>4</Size></Contents>\
            <Contents><Key>2010/06/01/fivan/201006010005_fivan_PVOL.h5</Key></Contents>\
            <NextContinuationToken>abc+/=</NextContinuationToken></ListBucketResult>";
        assert_eq!(xml_values(xml, "Key").len(), 2);
        assert_eq!(xml_values(xml, "IsTruncated"), vec!["true"]);
        assert_eq!(percent_encode("abc+/="), "abc%2B%2F%3D");
    }

    #[test]
    fn fmi_keys_of_another_site_or_product_are_not_frames() {
        let keys = vec![
            "2010/06/01/fivan/201006010005_fivan_PVOL.h5".to_string(),
            "2010/06/01/fivan/201006010000_fivan_PVOL.h5".to_string(),
            "2010/06/01/fivan/201006010000_fivan_SCAN.h5".to_string(),
            "2010/06/01/fikor/201006010000_fikor_PVOL.h5".to_string(),
        ];
        let plans = fmi_plans_from_keys(&keys, "fivan");
        assert_eq!(plans.len(), 2);
        assert_eq!(plans[0].identity, "201006010000_fivan_PVOL.h5");
        assert!(
            plans[1].parts[0]
                .url
                .ends_with("/2010/06/01/fivan/201006010005_fivan_PVOL.h5")
        );
    }

    #[test]
    fn dmi_items_become_oldest_first_plans_and_a_hrefless_item_is_refused() {
        let json = r#"{"features":[
            {"id":"dkste_202605010005.vol.h5","asset":{"data":{"href":"https://d/dkste_202605010005.vol.h5"}}},
            {"id":"dkste_202605010000.vol.h5","assets":{"data":{"href":"https://d/dkste_202605010000.vol.h5"}}}]}"#;
        let plans = dmi_plans_from_items(json).unwrap();
        assert_eq!(plans[0].identity, "dkste_202605010000.vol.h5");
        assert_eq!(plans.len(), 2);
        assert!(dmi_plans_from_items(r#"{"features":[{"id":"x"}]}"#).is_err());
    }
}
