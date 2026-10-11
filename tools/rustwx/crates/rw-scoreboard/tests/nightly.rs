use serde_json::{json,Value};
use std::{fs,path::{Path,PathBuf},process::Command};
// Parallel tests on Windows can read the same SystemTime nanoseconds; the counter keeps their scratch folders apart.
static NEXT_SCRATCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
struct Scratch(PathBuf);
impl Scratch { fn new()->Self {
    let p=PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("nightly-{}-{}-{}",std::process::id(),NEXT_SCRATCH.fetch_add(1,std::sync::atomic::Ordering::Relaxed),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    fs::create_dir_all(&p).unwrap();Self(p)
}}
impl Drop for Scratch {fn drop(&mut self){fs::remove_dir_all(&self.0).unwrap();}}
fn write(path:&Path,value:&Value){fs::write(path,serde_json::to_vec(value).unwrap()).unwrap();}
fn artifact(value:f64,init:Option<&str>)->Value {
    json!({"schema":"woof.scoreboard.artifact.v1","method_id":"scoreboard-v1","valid_time":"2026-10-07T06:00:00Z","init_time":init,"domain":"d01","lead_minutes":360,"lead_seconds":21600,"source_sha256":"fixture","support_hash":"frozen-support","stations":[{"station_id":"TEST","lat":40.,"lon":-100.,"elevation_m":0.}],"points":[{"station_id":"TEST","valid_time":"2026-10-07T06:00:00Z","values":{"temperature_2m":value,"dewpoint_2m":280.,"wind_speed_10m":4.,"u_wind_10m":value-290.,"v_wind_10m":0.}}],"grids":[]})
}
fn fixture(p:&Path)->Value {
    write(&p.join("forecast.json"),&artifact(291.,Some("2026-10-07T00:00:00Z")));
    write(&p.join("baseline.json"),&artifact(293.,Some("2026-10-07T00:00:00Z")));
    write(&p.join("observations.json"),&artifact(290.,None));
    let request=json!({"forecast":p.join("forecast.json"),"observations":p.join("observations.json"),"baseline":p.join("baseline.json")});
    json!({"schema":"woof.scoreboard.nightly-catalog.v1","complete":true,"window_start":"2026-09-08T12:00:00Z","window_end":"2026-10-08T12:00:00Z","comparator":"reference-model","runs":[{"run_id":"public_run","is_public":true,"deleted_at":null,"template":"generic-template","status":"finished","init_time":"2026-10-07T00:00:00Z","observations_final_after":"2026-10-08T06:00:00Z","frames":[{"valid_time":"2026-10-07T06:00:00Z","domain":"d01","baseline_model":"reference-model","score_request":request}]}]})
}
fn run(p:&Path,catalog:&Value)->std::process::Output {
    write(&p.join("catalog.json"),catalog);
    write(&p.join("request.json"),&json!({"action":"nightly","catalog":p.join("catalog.json"),"as_of":"2026-10-08T12:00:00Z","output_dir":p.join("out")}));
    Command::new(env!("CARGO_BIN_EXE_rw_scoreboard")).args(["--request",p.join("request.json").to_str().unwrap()]).output().unwrap()
}
#[test]
fn real_cli_identity_pooling_and_revision() {
    let scratch=Scratch::new();let p=&scratch.0;let catalog=fixture(p);
    let first=run(p,&catalog);assert!(first.status.success(),"{}",String::from_utf8_lossy(&first.stderr));
    let publication=fs::read(p.join("out/nightly.json")).unwrap();
    let modified=fs::metadata(p.join("out/nightly.json")).unwrap().modified().unwrap();
    let second=run(p,&catalog);assert!(second.status.success());assert_eq!(first.stdout,second.stdout);
    assert_eq!(publication,fs::read(p.join("out/nightly.json")).unwrap());
    assert_eq!(modified,fs::metadata(p.join("out/nightly.json")).unwrap().modified().unwrap());
    assert!(String::from_utf8_lossy(&second.stderr).contains("reused: 1"));
    let v:Value=serde_json::from_slice(&publication).unwrap();assert_eq!(v["seven_day"]["scored"],1);
    let t=v["templates"].as_array().unwrap().iter().find(|t|t["variable"]=="t2" && t["metric"]=="rmse").unwrap();
    assert_eq!(t["forecast"],1.);assert_eq!(t["comparator"],3.);
    let wind=v["templates"].as_array().unwrap().iter().find(|t|t["variable"]=="wind10" && t["metric"]=="rmse").unwrap();
    assert_eq!(wind["n"],1);assert_eq!(wind["forecast"],0.);assert_eq!(wind["comparator"],0.);
    write(&p.join("forecast.json"),&artifact(292.,Some("2026-10-07T00:00:00Z")));
    let third=run(p,&catalog);assert!(third.status.success());assert_ne!(first.stdout,third.stdout);
    assert!(String::from_utf8_lossy(&third.stderr).contains("reused: 0"));
}
#[test]
fn eligibility_finality_and_bad_catalog_are_not_silent() {
    let scratch=Scratch::new();let p=&scratch.0;let mut catalog=fixture(p);
    let mut private=catalog["runs"][0].clone();private["run_id"]=json!("private_run");private["is_public"]=json!(false);
    catalog["runs"].as_array_mut().unwrap().push(private);
    catalog["runs"][0]["observations_final_after"]=json!("2026-10-09T00:00:00Z");
    let output=run(p,&catalog);assert!(output.status.success());
    let v:Value=serde_json::from_slice(&output.stdout).unwrap();assert_eq!(v["seven_day"],json!({"scored":0,"total":1}));assert_eq!(v["runs"].as_array().unwrap().len(),1);
    assert_eq!(v["runs"][0]["reasons"][0],"observations_not_final");
    catalog["complete"]=json!(false);assert!(!run(p,&catalog).status.success());
    catalog["complete"]=json!(true);catalog["runs"][0]["run_id"]=json!("../escape");assert!(!run(p,&catalog).status.success());
}
#[test]
fn no_baseline_and_corrupt_cache_recover() {
    let scratch=Scratch::new();let p=&scratch.0;let mut catalog=fixture(p);
    assert!(run(p,&catalog).status.success());
    // Find the exact cache name rather than assuming a calendar conversion.
    let cache=fs::read_dir(p.join("out/runs")).unwrap().next().unwrap().unwrap().path();
    let mut value:Value=serde_json::from_slice(&fs::read(&cache).unwrap()).unwrap();value["score"]["pairs"]=json!([]);write(&cache,&value);
    let restored=run(p,&catalog);assert!(restored.status.success());assert!(String::from_utf8_lossy(&restored.stderr).contains("reused: 0"));
    fs::write(&cache,b"{truncated-cache").unwrap();
    let restored=run(p,&catalog);assert!(restored.status.success());assert!(String::from_utf8_lossy(&restored.stderr).contains("reused: 0"));
    catalog["runs"][0]["frames"][0]["score_request"]["baseline"]=Value::Null;
    let output=run(p,&catalog);assert!(output.status.success());let v:Value=serde_json::from_slice(&output.stdout).unwrap();assert_eq!(v["seven_day"]["scored"],0);
}
#[test]
fn a_different_frozen_method_is_reported_without_relabeling_scores() {
    let scratch=Scratch::new();let p=&scratch.0;let catalog=fixture(p);
    for name in ["forecast","baseline","observations"] {
        let path=p.join(format!("{name}.json"));
        let mut a:Value=serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        a["method_id"]=json!("scoreboard-v2");write(&path,&a);
    }
    let output=run(p,&catalog);assert!(output.status.success());
    let value:Value=serde_json::from_slice(&output.stdout).unwrap();assert_eq!(value["seven_day"]["scored"],0);
    assert!(value["runs"][0]["reasons"].to_string().contains("different frozen QC"));
}
fn gridded(value:f64,init:Option<&str>,rain60:Option<f64>,rain360:Option<f64>)->Value {
    let mut a=artifact(value,init);
    let spec=json!({"south":40.,"west":-100.,"dlat":0.01,"dlon":0.01,"nx":2,"ny":2,"dx_m":1000.});
    let mut grids=vec![];
    if let Some(r)=rain60 {grids.push(json!({"spec":spec,"variable":"precipitation","window_minutes":60,"values":[r,0.,0.,r]}));}
    if let Some(r)=rain360 {grids.push(json!({"spec":spec,"variable":"precipitation","window_minutes":360,"values":[r,r,0.,0.]}));}
    a["grids"]=json!(grids);a
}
fn two_request_fixture(p:&Path)->Value {
    // The single-request fixture first: it writes the plain artifacts these gridded ones replace.
    let mut catalog=fixture(p);
    write(&p.join("forecast.json"),&gridded(291.,Some("2026-10-07T00:00:00Z"),Some(2.),None));
    write(&p.join("forecast6.json"),&gridded(291.,Some("2026-10-07T00:00:00Z"),None,Some(8.)));
    write(&p.join("baseline.json"),&gridded(293.,Some("2026-10-07T00:00:00Z"),Some(0.),None));
    write(&p.join("baseline6.json"),&gridded(293.,Some("2026-10-07T00:00:00Z"),None,Some(0.)));
    write(&p.join("observations.json"),&gridded(290.,None,Some(2.),Some(8.)));
    let first=json!({"forecast":p.join("forecast.json"),"observations":p.join("observations.json"),"baseline":p.join("baseline.json")});
    let six=json!({"forecast":p.join("forecast6.json"),"observations":p.join("observations.json"),"baseline":p.join("baseline6.json"),"variable":"precipitation"});
    let frame=&mut catalog["runs"][0]["frames"][0];
    frame.as_object_mut().unwrap().remove("score_request");
    frame["score_requests"]=json!([first,six]);
    catalog["runs"][0]["frames"].as_array_mut().unwrap().push(json!({"valid_time":"2026-10-07T07:00:00Z","domain":"d01","baseline_model":"reference-model","skip_reason":"comparator_lead_unavailable"}));
    catalog
}
#[test]
fn two_rain_windows_one_frame_without_double_counting() {
    let scratch=Scratch::new();let p=&scratch.0;let catalog=two_request_fixture(p);
    let output=run(p,&catalog);assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    let v:Value=serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(v["schema"],"woof.scoreboard.nightly.v2");
    let rows=v["templates"].as_array().unwrap();
    let t2=rows.iter().find(|t|t["variable"]=="t2" && t["metric"]=="rmse").unwrap();
    // The six-hour request selects rain only: temperature is pooled once.
    assert_eq!(t2["n"],1);assert_eq!(t2["forecast"],1.);assert_eq!(t2["comparator"],3.);
    let ets=|w:u64,th:f64|rows.iter().find(|t|t["variable"]=="precipitation" && t["metric"]=="ets" && t["window_minutes"]==w && t["threshold"].as_f64()==Some(th)).cloned();
    let one=ets(60,1.).expect("hourly ETS pooled");assert_eq!(one["n"],4);assert_eq!(one["forecast"],1.);
    let six=ets(360,5.).expect("six-hour ETS pooled");assert_eq!(six["forecast"],1.);
    // HRRR-shaped baseline missed every event: no hits, so its ETS is below WOOF's.
    assert!(six["comparator"].as_f64().unwrap()<1.);
    // The skipped frame is named and counted as expected but not scored.
    let run_row=&v["runs"][0];
    assert_eq!(run_row["frames_total"],2);assert_eq!(run_row["frames_scored"],1);assert_eq!(run_row["status"],"partial");
    assert!(run_row["reasons"].to_string().contains("comparator_lead_unavailable"));
    // The run's own summary carries its pooled table and headline leads, and its hash is published.
    let summary_bytes=fs::read(p.join("out/summaries/public_run.json")).unwrap();
    use sha2::Digest;let hash=format!("{:x}",sha2::Sha256::digest(&summary_bytes));
    assert_eq!(run_row["summary_sha256"],hash);
    let summary:Value=serde_json::from_slice(&summary_bytes).unwrap();
    assert!(summary["scores"].as_array().unwrap().iter().any(|s|s["variable"]=="precipitation" && s["window_minutes"]==360));
    let lead=summary["lead_series"].as_array().unwrap();
    assert!(lead.iter().any(|l|l["variable"]=="t2" && l["metric"]=="rmse" && l["lead_seconds"]==json!([21600]) && l["forecast"]==json!([1.]) && l["comparator"]==json!([3.])));
    // The six-hour rain headline is a series of its own, at the 5 mm threshold.
    assert!(lead.iter().any(|l|l["variable"]=="precipitation" && l["window_minutes"]==360 && l["threshold"].as_f64()==Some(5.)));
    assert!(v["template_day_series"].as_array().unwrap().iter().any(|d|d["days"]==json!(["2026-10-07"]) && d["variable"]=="t2" && d["metric"]=="rmse" && d["run_count"]==json!([1])));
    assert!(v["template_lead_series"].as_array().unwrap().iter().all(|d|d["lead_seconds"]==json!([21600])));
    assert_eq!(v["thirty_day"],json!({"scored":1,"total":1}));
}
#[test]
fn retired_inputs_reuse_the_verified_receipt_and_bytes() {
    let scratch=Scratch::new();let p=&scratch.0;let mut catalog=two_request_fixture(p);
    assert!(run(p,&catalog).status.success());
    let publication=fs::read(p.join("out/nightly.json")).unwrap();
    for name in ["forecast","forecast6","baseline","baseline6","observations"] {fs::remove_file(p.join(format!("{name}.json"))).unwrap();}
    catalog["runs"][0]["frames"][0]["inputs_retired"]=json!(true);
    let again=run(p,&catalog);assert!(again.status.success(),"{}",String::from_utf8_lossy(&again.stderr));
    assert_eq!(publication,fs::read(p.join("out/nightly.json")).unwrap());
    assert!(String::from_utf8_lossy(&again.stderr).contains("reused: 2"));
    // A retired frame whose receipt was tampered with is a named gap, never a silent score.
    let cache=p.join("out/runs").read_dir().unwrap().map(|e|e.unwrap().path()).find(|p|!p.to_string_lossy().ends_with("-1.json")).unwrap();
    let mut value:Value=serde_json::from_slice(&fs::read(&cache).unwrap()).unwrap();value["score"]["pairs"]=json!([]);write(&cache,&value);
    let tampered=run(p,&catalog);assert!(tampered.status.success());
    let v:Value=serde_json::from_slice(&tampered.stdout).unwrap();
    assert!(v["runs"][0]["reasons"].to_string().contains("inputs_retired_without_verified_receipt"));
}
#[test]
fn overlapping_requests_in_one_frame_are_named_not_double_counted() {
    let scratch=Scratch::new();let p=&scratch.0;let mut catalog=two_request_fixture(p);
    // The second request now also carries the hourly window: that overlap must not pool twice.
    write(&p.join("forecast6.json"),&gridded(291.,Some("2026-10-07T00:00:00Z"),Some(2.),Some(8.)));
    write(&p.join("baseline6.json"),&gridded(293.,Some("2026-10-07T00:00:00Z"),Some(0.),Some(0.)));
    catalog["runs"][0]["frames"].as_array_mut().unwrap().truncate(1);
    let output=run(p,&catalog);assert!(output.status.success());
    let v:Value=serde_json::from_slice(&output.stdout).unwrap();
    let one=v["templates"].as_array().unwrap().iter().find(|t|t["variable"]=="precipitation" && t["window_minutes"]==60 && t["threshold"].as_f64()==Some(1.)).cloned().unwrap();
    assert_eq!(one["n"],4);
    assert!(v["runs"][0]["reasons"].to_string().contains("overlapping_score_requests"));
}
