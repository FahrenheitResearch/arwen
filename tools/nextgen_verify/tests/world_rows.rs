// SPDX-License-Identifier: Apache-2.0
//! Worldwide gauge rows decode through the same normalize path as USGS.
//! Documents are shaped like the services' real responses (PEGELONLINE zip
//! document after container decode, ECCC OGC API features, BOM KiWIS rows).
use nextgen_verify::*;
use serde_json::json;

fn table() -> Table {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../gpuwm/hydro/usgs_table.json");
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
fn row<'a>(t: &'a Table, id: &str) -> &'a Source {
    t.sources.iter().find(|s| s.id == id).unwrap_or_else(|| panic!("no source row {id}"))
}
fn station(id: &str, source: &str) -> Station {
    Station { station_id: id.into(), source: Some(source.into()), ..Default::default() }
}

#[test]
fn pegelonline_archive_document_uses_request_station_and_document_unit() {
    let t = table();
    let s = row(&t, "pegelonline-archive");
    let doc = json!({"records":[{"timestamp":"2023-01-01T01:00:00+01:00","value":2940},{"timestamp":"2023-01-01T01:15:00+01:00","value":2930}],
        "info":{"timeseries_unit":"m³/s","quality":"unchecked"}});
    let (rows, rejected) = normalize(&doc, &s.layout, &[station("PEGEL:2730010", "pegelonline-archive")]).unwrap();
    assert_eq!((rows.len(), rejected), (2, 0));
    assert_eq!(rows[0].station_id, "PEGEL:2730010");
    assert_eq!(rows[0].time, "2023-01-01T00:00:00+00:00");
    assert_eq!(rows[0].flow_m3s, 2940.0);
    assert!(rows[0].quality.contains("unchecked"));
}

#[test]
fn eccc_daily_means_need_an_explicit_offset_and_skip_missing_values() {
    let t = table();
    let s = row(&t, "eccc-hydat-daily");
    let doc = json!({"features":[
        {"properties":{"STATION_NUMBER":"05BH004","DATE":"2023-01-01","DISCHARGE":43.79999923706055,"DISCHARGE_SYMBOL_EN":"Ice Conditions"}},
        {"properties":{"STATION_NUMBER":"05BH004","DATE":"2023-01-02","DISCHARGE":null,"DISCHARGE_SYMBOL_EN":null}}]});
    let mut st = station("WSC:05BH004", "eccc-hydat-daily");
    assert!(normalize(&doc, &s.layout, std::slice::from_ref(&st)).unwrap_err().to_string().contains("utc_offset"));
    st.utc_offset = Some("-07:00".into());
    let (rows, rejected) = normalize(&doc, &s.layout, &[st]).unwrap();
    assert_eq!((rows.len(), rejected), (1, 1));
    assert_eq!(rows[0].time, "2023-01-01T07:00:00+00:00");
    assert!(rows[0].quality.contains("Ice Conditions"));
}

#[test]
fn bom_rows_convert_units_and_keep_quality_codes() {
    let t = table();
    let s = row(&t, "bom-wdo");
    let doc = json!([{"ts_unitsymbol":"cumec","station_no":"410001","DATA_OWNER":"w00077",
        "data":[["2023-01-01T10:00:00.000+10:00",188.298,140],["2023-01-01T11:00:00.000+10:00",188.921,140]]}]);
    let (rows, rejected) = normalize(&doc, &s.layout, &[station("BOM:410001", "bom-wdo")]).unwrap();
    assert_eq!((rows.len(), rejected), (2, 0));
    assert_eq!(rows[0].time, "2023-01-01T00:00:00+00:00");
    assert_eq!(rows[0].flow_m3s, 188.298);
    assert!(rows[0].quality.contains("140"));
    let gate = s.owner_gate.as_ref().unwrap();
    assert!(world::owner_attribution(gate, &doc, "BOM:410001").is_ok());
    let snowy = json!([{"DATA_OWNER":"w00205","data":[]}]);
    assert!(world::owner_attribution(gate, &snowy, "BOM:x").unwrap_err().to_string().contains("EXCLUDED"));
}

#[test]
fn pointer_member_filter_picks_the_discharge_series() {
    let meta = json!({"uuid":"u","timeseries":[{"shortname":"W","longname":"WASSERSTAND ROHDATEN"},{"shortname":"Q","longname":"ABFLUSS"}]});
    assert_eq!(field(&meta, "/timeseries/[shortname=Q]/longname").as_deref(), Some("ABFLUSS"));
    assert_eq!(field(&meta, "/timeseries/[shortname=X]/longname"), None);
}

#[test]
fn excluded_services_refuse_with_their_licence_clause() {
    let t = table();
    for id in ["uk-nrfa", "grdc"] {
        let e = world::admit(row(&t, id)).unwrap_err().to_string();
        assert!(e.contains("EXCLUDED") && e.len() > 80, "{id}: {e}");
    }
    for s in &t.sources {
        if s.licence.as_ref().is_some_and(|l| l.status == "allowed") {
            world::admit(s).unwrap();
        }
    }
}
