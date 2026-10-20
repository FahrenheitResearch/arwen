// SPDX-License-Identifier: Apache-2.0
//! Worldwide gauge rows: the parts of a source row beyond the original USGS
//! layout. Every national service is described by data in the gauge table
//! (request shape, container, station metadata, licence and owner gate); the
//! fetcher has no per-agency code path.
use crate::{field, select, Layout, Result, Source, Station};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::io::Read;

pub fn empty_layout() -> Layout {
    Layout {
        records: String::new(),
        station: None,
        station_prefix: String::new(),
        feature: None,
        time: String::new(),
        flow: String::new(),
        unit: None,
        fixed_unit: None,
        unit_factors: BTreeMap::new(),
        filters: BTreeMap::new(),
        optional_filters: BTreeMap::new(),
        time_series: None,
        netcdf: None,
        station_from_request: false,
        quality: None,
        fixed_quality: None,
        time_from_date: false,
    }
}

/// Licence verdict carried by a source row, copied from the licence lane's
/// table (docs/public/HYDRO-WORLD-LICENCES.md) with the operative sentence.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Licence {
    /// `allowed` or `excluded`. An excluded row is never fetched.
    pub status: String,
    pub licence: String,
    #[serde(default)]
    pub attribution: String,
    /// The operative sentence, verbatim, as read at the source.
    pub quote: String,
    /// Where the verdict and the saved licence bytes are recorded.
    pub evidence: String,
}

/// How a data request is made. Default: GET of `url`, paginated by `links`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Request {
    /// `GET` or `POST_FORM`.
    pub method: String,
    /// Form fields (templates) for `POST_FORM`.
    #[serde(default)]
    pub form: BTreeMap<String, String>,
    /// Follow one redirect to the same origin (the POST-then-download pattern).
    #[serde(default)]
    pub follow_same_origin_redirect: bool,
    /// Optional User-Agent the service requires.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
    /// Requests in flight against this service (default 8). Services that build
    /// a file per request (the PEGELONLINE archive) answer an error page when crowded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_in_flight: Option<usize>,
}

/// Response container. Default: the body is one JSON document.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Container {
    /// `zip`: members are decoded into {"records": <json member>, "info": {...}}.
    pub kind: String,
    /// The JSON member holding the records is the one with this suffix.
    pub records_member_suffix: String,
    /// Optional `key=value` text member, exposed as `info`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info_member: Option<String>,
}

/// Station metadata request: location, name, drainage area, and any extra
/// values later templates need (for example a service-internal uuid).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Metadata {
    pub url: String,
    pub latitude: String,
    pub longitude: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drainage_area: Option<String>,
    /// Multiplier from the source's drainage-area unit to km2.
    #[serde(default = "one")]
    pub drainage_area_to_km2: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drainage_area_unit_note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
}
fn one() -> f64 {
    1.0
}

/// Per-owner licence gate (BOM Water Data Online: each data owner sets its
/// own terms). A station whose owner is not listed is refused, by name.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OwnerGate {
    /// Pointer to the owner code (document-level with `^`).
    pub owner: String,
    /// Allowed owner code -> attribution line.
    pub allowed: BTreeMap<String, String>,
    /// Owner codes known to be excluded -> reason.
    #[serde(default)]
    pub excluded: BTreeMap<String, String>,
    pub source_of_list: String,
}

/// Station discovery inside a bounding box: table work, not code.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Discover {
    /// URL; `{bbox}` = minlon,minlat,maxlon,maxlat when the service filters, else the full list.
    pub url: String,
    pub records: String,
    pub id: String,
    pub latitude: String,
    pub longitude: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Wildcard pointer whose values must include `require_value` (for example a discharge series).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
}

/// The common station schema every source row produces.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct StationInfo {
    pub station_id: String,
    pub agency: String,
    pub source: String,
    pub name: Option<String>,
    pub latitude: f64,
    pub longitude: f64,
    pub drainage_area_km2: Option<f64>,
    /// `source metadata`, `table row: <citation>`, or `not published by source`.
    pub drainage_area_origin: String,
    /// Contract hydrolocation uri: `Gages-<n>` (USGS) or `Gages-<AGENCY>:<n>`.
    pub hl_uri: String,
    pub licence: String,
    pub attribution: String,
}

pub fn hl_uri(station_id: &str) -> String {
    format!("Gages-{station_id}")
}

/// Refuse an excluded or verdict-less row, naming the clause that rules it out.
pub fn admit(source: &Source) -> Result<()> {
    if source.agency.is_none() && source.licence.is_none() {
        // The original USGS row predates licence blocks; USGS data is US public domain.
        return Ok(());
    }
    let Some(l) = &source.licence else {
        return Err(format!("source {} has no licence verdict; it is not fetched", source.id).into());
    };
    match l.status.as_str() {
        "allowed" => Ok(()),
        "excluded" => Err(format!(
            "source {} is EXCLUDED by its licence ({}): \"{}\" (verdict: {})",
            source.id, l.licence, l.quote, l.evidence
        )
        .into()),
        other => Err(format!("source {} has licence status {other:?}; only allowed rows are fetched", source.id).into()),
    }
}

/// Does a station row belong to this source row?
pub fn belongs(station: &Station, source: &Source) -> bool {
    match &station.source {
        Some(s) => s == &source.id,
        None => source.agency.is_none(),
    }
}

/// Template expansion. `{station}` is the agency number, `{start}`/`{end}`
/// the RFC3339 window, `{meta:<pointer>}` a value from the metadata document.
/// `encode` percent-encodes substituted values (URLs); form fields are raw.
pub fn expand(
    template: &str,
    station: &Station,
    start: &str,
    end: &str,
    meta: Option<&Value>,
    encode: &dyn Fn(&str) -> String,
) -> Result<String> {
    let mut out = String::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let close = rest[open..].find('}').ok_or("unclosed template brace")? + open;
        let key = &rest[open + 1..close];
        let value = match key {
            "station" => station.number().to_string(),
            "start" => start.to_string(),
            "end" => end.to_string(),
            k if k.starts_with("meta:") => {
                let m = meta.ok_or("template uses {meta:...} but the row has no metadata request")?;
                field(m, &k[5..]).ok_or_else(|| format!("metadata has no value at {}", &k[5..]))?
            }
            other => return Err(format!("unknown template key {{{other}}}").into()),
        };
        out.push_str(&encode(&value));
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Decode a response body into the JSON document the layout reads.
pub fn decode(raw: &[u8], container: Option<&Container>) -> Result<Value> {
    let Some(c) = container else {
        return Ok(serde_json::from_slice(raw)?);
    };
    if c.kind != "zip" {
        return Err(format!("unknown container kind {}", c.kind).into());
    }
    let members = unzip(raw)?;
    let mut doc = Map::new();
    let records: Vec<_> = members
        .iter()
        .filter(|(n, _)| n.ends_with(&c.records_member_suffix))
        .collect();
    if records.len() != 1 {
        return Err(format!(
            "zip must hold exactly one member ending {:?}; found {:?}",
            c.records_member_suffix,
            members.keys().collect::<Vec<_>>()
        )
        .into());
    }
    doc.insert("records".into(), serde_json::from_slice(records[0].1)?);
    if let Some(info) = &c.info_member {
        let text = members
            .get(info)
            .ok_or_else(|| format!("zip has no member {info}"))?;
        let mut kv = Map::new();
        for line in String::from_utf8_lossy(text).lines() {
            if line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                kv.insert(k.trim().into(), Value::String(v.trim().into()));
            }
        }
        doc.insert("info".into(), Value::Object(kv));
    }
    doc.insert(
        "members".into(),
        Value::Array(members.keys().map(|k| Value::String(k.clone())).collect()),
    );
    Ok(Value::Object(doc))
}

/// Minimal ZIP reader (central directory, stored or deflate members).
pub fn unzip(raw: &[u8]) -> Result<BTreeMap<String, Vec<u8>>> {
    let u16le = |o: usize| -> Result<usize> {
        raw.get(o..o + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
            .ok_or_else(|| "truncated zip".into())
    };
    let u32le = |o: usize| -> Result<usize> {
        raw.get(o..o + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
            .ok_or_else(|| "truncated zip".into())
    };
    let eocd = (0..raw.len().saturating_sub(21))
        .rev()
        .find(|&i| raw[i..].starts_with(b"PK\x05\x06"))
        .ok_or("not a zip archive (no end of central directory)")?;
    let count = u16le(eocd + 10)?;
    let mut at = u32le(eocd + 16)?;
    let mut out = BTreeMap::new();
    for _ in 0..count {
        if u32le(at)? != 0x0201_4b50 {
            return Err("bad zip central directory".into());
        }
        let method = u16le(at + 10)?;
        let csize = u32le(at + 20)?;
        let usize_ = u32le(at + 24)?;
        let nlen = u16le(at + 28)?;
        let elen = u16le(at + 30)?;
        let clen = u16le(at + 32)?;
        let local = u32le(at + 42)?;
        let name = String::from_utf8_lossy(raw.get(at + 46..at + 46 + nlen).ok_or("truncated zip")?).into_owned();
        at += 46 + nlen + elen + clen;
        if u32le(local)? != 0x0403_4b50 {
            return Err("bad zip local header".into());
        }
        let data_at = local + 30 + u16le(local + 26)? + u16le(local + 28)?;
        let data = raw.get(data_at..data_at + csize).ok_or("truncated zip member")?;
        let bytes = match method {
            0 => data.to_vec(),
            8 => {
                let mut v = Vec::with_capacity(usize_);
                flate2::read::DeflateDecoder::new(data).read_to_end(&mut v)?;
                v
            }
            m => return Err(format!("zip member {name} uses unsupported method {m}").into()),
        };
        if bytes.len() != usize_ {
            return Err(format!("zip member {name} size mismatch").into());
        }
        out.insert(name, bytes);
    }
    Ok(out)
}

fn number(v: Option<String>) -> Option<f64> {
    v.and_then(|s| s.trim().parse::<f64>().ok()).filter(|x| x.is_finite())
}

/// Build the common station record from the metadata document and the row.
pub fn station_info(source: &Source, station: &Station, meta: Option<&Value>) -> Result<StationInfo> {
    let m = source.metadata.as_ref();
    let get = |p: Option<&String>| -> Option<String> { meta.zip(p).and_then(|(d, p)| field(d, p)) };
    let latitude = station
        .latitude
        .or_else(|| number(get(m.map(|m| &m.latitude))))
        .ok_or_else(|| format!("station {}: no latitude in row or metadata", station.station_id))?;
    let longitude = station
        .longitude
        .or_else(|| number(get(m.map(|m| &m.longitude))))
        .ok_or_else(|| format!("station {}: no longitude in row or metadata", station.station_id))?;
    if !(-90.0..=90.0).contains(&latitude) || !(-180.0..=180.0).contains(&longitude) {
        return Err(format!("station {}: location out of range", station.station_id).into());
    }
    let (area, origin) = if let Some(a) = station.drainage_area_km2 {
        let cite = station
            .drainage_area_source
            .clone()
            .ok_or_else(|| format!("station {}: drainage_area_km2 needs drainage_area_source", station.station_id))?;
        (Some(a), format!("table row: {cite}"))
    } else if let Some(a) = number(get(m.and_then(|m| m.drainage_area.as_ref()))).filter(|a| *a > 0.) {
        let f = m.map_or(1.0, |m| m.drainage_area_to_km2);
        let note = m.and_then(|m| m.drainage_area_unit_note.clone()).unwrap_or_default();
        (Some(a * f), format!("source metadata {note}").trim().to_string())
    } else {
        (None, "not published by source".to_string())
    };
    let name = station
        .name
        .clone()
        .or_else(|| get(m.and_then(|m| m.name.as_ref())));
    let licence = source.licence.as_ref();
    Ok(StationInfo {
        station_id: station.station_id.clone(),
        agency: source.agency.clone().unwrap_or_else(|| "USGS".into()),
        source: source.id.clone(),
        name,
        latitude,
        longitude,
        drainage_area_km2: area,
        drainage_area_origin: origin,
        hl_uri: hl_uri(&station.station_id),
        licence: licence.map_or("US public domain".into(), |l| l.licence.clone()),
        attribution: licence.map_or("Credit: U.S. Geological Survey (NWIS)".into(), |l| l.attribution.clone()),
    })
}

/// Apply the per-owner gate to one data document. Returns the attribution line.
pub fn owner_attribution(gate: &OwnerGate, document: &Value, station: &str) -> Result<String> {
    let owner = crate::lookup(document, document, &gate.owner)
        .ok_or_else(|| format!("station {station}: no data owner at {}; refused (owner terms unknown)", gate.owner))?;
    if let Some(a) = gate.allowed.get(&owner) {
        return Ok(format!("{owner}: {a}"));
    }
    let why = gate
        .excluded
        .get(&owner)
        .cloned()
        .unwrap_or_else(|| "owner not on the CC BY list".into());
    Err(format!(
        "station {station}: data owner {owner} is EXCLUDED ({why}); list: {}",
        gate.source_of_list
    )
    .into())
}

/// Parse a discovery document into station rows inside the box.
pub fn discovered(spec: &Discover, source: &Source, doc: &Value, bbox: [f64; 4]) -> Vec<Station> {
    let mut rows = Vec::new();
    for r in select(doc, &spec.records) {
        let Some(id) = field(r, &spec.id) else { continue };
        let (Some(lat), Some(lon)) = (number(field(r, &spec.latitude)), number(field(r, &spec.longitude))) else {
            continue;
        };
        if lon < bbox[0] || lat < bbox[1] || lon > bbox[2] || lat > bbox[3] {
            continue;
        }
        if let (Some(p), Some(want)) = (&spec.require, &spec.require_value) {
            if !select(r, p).iter().any(|v| crate::scalar(v).as_deref() == Some(want.as_str())) {
                continue;
            }
        }
        let station_id = match &source.agency {
            Some(a) => format!("{a}:{id}"),
            None => id,
        };
        rows.push(Station {
            station_id,
            feature_id: None,
            time_series_id: None,
            source: Some(source.id.clone()),
            utc_offset: None,
            name: spec.name.as_ref().and_then(|p| field(r, p)),
            latitude: Some(lat),
            longitude: Some(lon),
            drainage_area_km2: None,
            drainage_area_source: None,
        });
    }
    rows.sort_by(|a, b| a.station_id.cmp(&b.station_id));
    rows.dedup_by(|a, b| a.station_id == b.station_id);
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    fn station(id: &str) -> Station {
        serde_json::from_value(serde_json::json!({"station_id": id})).unwrap()
    }
    #[test]
    fn templates_expand_agency_number_and_metadata() {
        let s = station("PEGEL:2730010");
        let meta = serde_json::json!({"uuid": "a6ee"});
        let id = |v: &str| v.to_string();
        let out = expand("{station}/{meta:/uuid}/{start}", &s, "2023-01-01T00:00:00Z", "", Some(&meta), &id).unwrap();
        assert_eq!(out, "2730010/a6ee/2023-01-01T00:00:00Z");
        assert!(expand("{nope}", &s, "", "", None, &id).is_err());
        assert!(expand("{meta:/uuid}", &s, "", "", None, &id).is_err());
    }
    #[test]
    fn zip_members_decode_into_records_and_info() {
        // Stored (method 0) archive built by hand: one json member and one info member.
        fn member(name: &str, body: &[u8], offset: u32) -> (Vec<u8>, Vec<u8>) {
            let mut local = b"PK\x03\x04".to_vec();
            local.extend([20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            local.extend((body.len() as u32).to_le_bytes());
            local.extend((body.len() as u32).to_le_bytes());
            local.extend((name.len() as u16).to_le_bytes());
            local.extend([0, 0]);
            local.extend(name.as_bytes());
            local.extend(body);
            let mut central = b"PK\x01\x02".to_vec();
            central.extend([20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            central.extend((body.len() as u32).to_le_bytes());
            central.extend((body.len() as u32).to_le_bytes());
            central.extend((name.len() as u16).to_le_bytes());
            central.extend([0; 12]);
            central.extend(offset.to_le_bytes());
            central.extend(name.as_bytes());
            (local, central)
        }
        let (l1, c1) = member("x-Q.json", br#"[{"timestamp":"2023-01-01T01:00:00+01:00","value":2940}]"#, 0);
        let (l2, c2) = member("info.txt", b"# note\ntimeseries_unit=m\xc2\xb3/s\nquality=unchecked\n", l1.len() as u32);
        let mut z = [l1.clone(), l2.clone()].concat();
        let cd = z.len() as u32;
        z.extend(&c1);
        z.extend(&c2);
        z.extend(b"PK\x05\x06\0\0\0\0\x02\0\x02\0");
        z.extend(((c1.len() + c2.len()) as u32).to_le_bytes());
        z.extend(cd.to_le_bytes());
        z.extend([0, 0]);
        let c = Container { kind: "zip".into(), records_member_suffix: ".json".into(), info_member: Some("info.txt".into()) };
        let doc = decode(&z, Some(&c)).unwrap();
        assert_eq!(doc.pointer("/records/0/value").unwrap(), 2940);
        assert_eq!(doc.pointer("/info/quality").unwrap(), "unchecked");
        assert_eq!(doc.pointer("/info/timeseries_unit").unwrap(), "m³/s");
    }
    #[test]
    fn owner_gate_refuses_unlisted_owner_by_name() {
        let gate = OwnerGate {
            owner: "^/0/DATA_OWNER".into(),
            allowed: BTreeMap::from([("w00077".into(), "© WaterNSW".into())]),
            excluded: BTreeMap::from([("w00205".into(), "licence other".into())]),
            source_of_list: "owners.tsv".into(),
        };
        let ok = serde_json::json!([{"DATA_OWNER": "w00077"}]);
        assert!(owner_attribution(&gate, &ok, "s").unwrap().contains("WaterNSW"));
        let bad = serde_json::json!([{"DATA_OWNER": "w00205"}]);
        let e = owner_attribution(&gate, &bad, "s").unwrap_err().to_string();
        assert!(e.contains("w00205") && e.contains("EXCLUDED"));
    }
}
