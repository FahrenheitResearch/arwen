// SPDX-License-Identifier: Apache-2.0
use nextgen_verify::*;
use serde_json::json;

fn row(station: &str, time: &str, q: f64) -> Observation {
    Observation {
        station_id: station.into(),
        time: time.into(),
        flow_m3s: q,
        quality: String::new(),
    }
}
#[test]
fn utc_offsets_join_and_conflicting_values_fail() {
    let a = row("g", "2023-01-05T00:00:00Z", 4.);
    let b = row("g", "2023-01-04T16:00:00-08:00", 4.);
    assert_eq!(ordered(&[a.clone(), b]).unwrap().len(), 1);
    assert!(index(&[a, row("g", "2023-01-04T16:00:00-08:00", 5.)]).is_err());
}
#[test]
fn scores_have_known_values_and_positive_late_peak() {
    let m = metrics(&[(0, 1., 2.), (1000, 2., 4.), (2000, 3., 6.)]);
    assert_eq!(m.nse, Some(-0.75));
    assert_eq!(m.mean_ratio, Some(0.5));
    assert_eq!(m.kge_2009, Some(1. - 0.5_f64.sqrt()));
    let p = metrics(&[(0, 1., 3.), (1000, 3., 1.), (2000, 3., 2.)]);
    assert_eq!(p.peak_timing_seconds, Some(1.));
    let one = metrics(&[(0, 1., 1.)]);
    assert!(one.nse.is_none() && one.kge_2009.is_none() && one.peak_timing_seconds.is_none());
    let dry = metrics(&[(0, 1., 0.), (1000, 2., 0.)]);
    assert!(dry.nse.is_none() && dry.kge_2009.is_none() && dry.peak_timing_seconds.is_none());
    assert!(dry.rmse_m3s.is_some());
    let flat_prediction = metrics(&[(0, 1., 1.), (1000, 1., 2.)]);
    assert!(flat_prediction.peak_timing_seconds.is_none());
}
#[test]
fn observation_scores_share_exact_three_way_support() {
    let station = "g";
    let make = |entries: &[(i64, f64)]| {
        std::collections::BTreeMap::from([(station.to_owned(), entries.iter().copied().collect())])
    };
    let report = assess(
        station,
        &make(&[(0, 1.), (1000, 2.), (2000, 100.)]),
        &make(&[(0, 1.), (1000, 2.)]),
        &make(&[(0, 1.), (2000, 50.)]),
    );
    assert!(!report.complete_reference_support);
    assert_eq!(report.model_vs_reference.samples, 2);
    assert_eq!(report.model_vs_observations.samples, 1);
    assert_eq!(report.reference_vs_observations.samples, 1);
}
#[test]
fn another_source_is_a_table_row_and_invalid_flows_are_rejected() {
    let layout:Layout=serde_json::from_value(json!({"records":"/measurements/*","feature":"/reach","time":"/utc","flow":"/q","unit":"/unit","unit_factors":{"cfs":0.028316846592},"filters":{"/status":["A"]},"optional_filters":{"/flag":["e"]}})).unwrap();
    let station = Station {
        station_id: "g".into(),
        feature_id: Some("42".into()),
        time_series_id: None,
        ..Default::default()
    };
    let root = json!({"measurements":[{"reach":42,"utc":"2023-01-05T00:00:00Z","q":1.,"unit":"cfs","status":"A","flag":"e"},{"reach":42,"utc":"2023-01-05T00:00:00Z","q":-999999.,"unit":"cfs","status":"A"},{"reach":42,"utc":"2023-01-05T00:00:00Z","q":1.,"unit":"cfs","status":"R"}]});
    let (rows, rejected) = normalize(&root, &layout, &[station]).unwrap();
    assert_eq!(rejected, 2);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].flow_m3s.to_bits(), 0.028316846592_f64.to_bits());
    assert!(rows[0].quality.contains('e'));
}

#[test]
fn executable_reports_a_missing_observed_station_instead_of_dropping_it() {
    let directory = std::env::temp_dir().join(format!("nextgen-score-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let time = "2023-01-05T00:00:00Z";
    let observed = vec![row("present", time, 1.), row("missing", time, 1.)];
    let model = vec![row("present", time, 1.)];
    let save = |name: &str, rows: &[Observation]| {
        let path = directory.join(name);
        std::fs::write(&path, serde_json::to_vec(rows).unwrap()).unwrap();
        path
    };
    let m = save("model.json", &model);
    let o = save("observed.json", &observed);
    let out = directory.join("score");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_rw_nextgen_verify"))
        .arg("score")
        .arg(&m)
        .arg(&m)
        .arg(&o)
        .arg(&out)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("scores.json")).unwrap()).unwrap();
    assert_eq!(receipt["stations"].as_array().unwrap().len(), 2);
    assert_eq!(receipt["complete_support_bitwise_f64"], false);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn executable_compares_a_labelled_prediction_on_exact_observation_support() {
    let directory = std::env::temp_dir().join(format!("nextgen-compare-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let predicted = vec![
        row("g", "2023-01-05T00:00:00Z", 2.),
        row("g", "2023-01-05T01:00:00Z", 4.),
    ];
    let observed = vec![
        row("g", "2023-01-04T16:00:00-08:00", 1.),
        row("g", "2023-01-05T00:15:00Z", 100.),
        row("g", "2023-01-05T01:00:00Z", 2.),
    ];
    let prediction = directory.join("prediction.json");
    let observations = directory.join("observations.json");
    std::fs::write(&prediction, serde_json::to_vec(&predicted).unwrap()).unwrap();
    std::fs::write(&observations, serde_json::to_vec(&observed).unwrap()).unwrap();
    let out = directory.join("report");
    let run = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_rw_nextgen_verify"))
            .arg("compare")
            .arg(&prediction)
            .arg("Archive <v3>")
            .arg(&observations)
            .arg(&out)
            .output()
            .unwrap()
    };
    assert!(run().status.success());
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("comparison.json")).unwrap()).unwrap();
    let report = &receipt["stations"][0];
    assert_eq!(report["scores"]["samples"], 2);
    assert_eq!(report["unmatched_observation_samples"], 1);
    assert_eq!(report["unmatched_prediction_samples"], 0);
    assert_eq!(report["scores"]["nse"], -9.);
    assert!(receipt.get("complete_support_bitwise_f64").is_none());
    let svg = std::fs::read_to_string(out.join(report["picture"].as_str().unwrap())).unwrap();
    assert!(svg.contains("Archive &lt;v3&gt;"));
    assert!(!svg.contains(">GPU<") && !svg.contains(">Native<"));
    let mut missing = observed;
    missing.push(row("missing", "2023-01-05T00:00:00Z", 1.));
    std::fs::write(&observations, serde_json::to_vec(&missing).unwrap()).unwrap();
    assert!(!run().status.success());
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("comparison.json")).unwrap()).unwrap();
    assert_eq!(receipt["stations"].as_array().unwrap().len(), 2);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn executable_keeps_native_equality_when_a_gauge_lacks_observations() {
    let directory = std::env::temp_dir().join(format!("nextgen-parity-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let time = "2023-01-05T00:00:00Z";
    let model = vec![row("observed", time, 1.), row("unobserved", time, 2.)];
    let m = directory.join("model.json");
    let o = directory.join("observations.json");
    std::fs::write(&m, serde_json::to_vec(&model).unwrap()).unwrap();
    std::fs::write(&o, serde_json::to_vec(&model[..1]).unwrap()).unwrap();
    let out = directory.join("scores");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_rw_nextgen_verify"))
        .args(["score"])
        .arg(&m)
        .arg(&m)
        .arg(&o)
        .arg(&out)
        .output()
        .unwrap();
    assert!(!result.status.success());
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("scores.json")).unwrap()).unwrap();
    assert_eq!(receipt["complete_support_bitwise_f64"], true);
    assert_eq!(receipt["all_stations_have_observation_pairs"], false);
    assert_eq!(receipt["stations"].as_array().unwrap().len(), 2);
    std::fs::remove_dir_all(directory).unwrap();
}
