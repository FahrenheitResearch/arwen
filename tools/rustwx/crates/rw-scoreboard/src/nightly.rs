//! Deterministic CPU scoring of an authoritative public-run catalog.
//!
//! One pass reads the catalog of every public run in the rolling 30-day window, scores each frame
//! that is final (or reuses its verified receipt), and publishes:
//! - `nightly.json`: run rows with status and named reasons, the 7- and 30-day counts, the per-template
//!   30-day table pooled across leads and runs, a per-lead and a per-initialization-day table for the
//!   headline metrics;
//! - `summaries/<run>.json`: each scored run's own pooled table and its per-lead headline rows.
//! Every table pools sufficient statistics; RMSE, ETS and FSS are never averaged.
use crate::*;
use rw_scoreboard::Stats;
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::path::PathBuf;

const SCHEMA: &str = "woof.scoreboard.nightly-catalog.v1";
const SUMMARY_SCHEMA: &str = "woof.scoreboard.run-summary.v1";

fn write(path: &Path, value: &Value) -> Result<String> {
    let bytes=serde_json::to_vec_pretty(value)?;
    let hash=digest(&bytes);
    if fs::read(path).ok().as_deref()==Some(bytes.as_slice()) { return Ok(hash); }
    let temporary=path.with_extension("tmp");
    fs::write(&temporary,&bytes)?;
    fs::rename(temporary,path)?;
    Ok(hash)
}
fn safe_id(id: &str) -> bool {
    !id.is_empty() && id.len()<=128 && id.bytes().all(|c|c.is_ascii_alphanumeric() || c==b'_' || c==b'-')
}
/// A number's identity in a key: 25 and 25.0 are one scale.
fn num(v: &Value) -> Value {
    v.as_f64().map_or(Value::Null, |f| json!(f))
}
fn key(pair: &Value) -> Result<String> {
    Ok(serde_json::to_string(&json!([pair["variable"],pair["windowMinutes"],pair["metric"],pair["units"],num(&pair["threshold"]),num(&pair["scaleKm"]),num(&pair["windowHeightKm"]),num(&pair["windowWidthMinKm"]),num(&pair["windowWidthMaxKm"])]))?)
}
fn wanted(pair:&Value) -> bool {
    let variable=pair["variable"].as_str().unwrap_or("");
    let metric=pair["metric"].as_str().unwrap_or("");
    (matches!(variable,"t2"|"td2"|"wind10") && matches!(metric,"rmse"|"bias")
        && (variable!="wind10" || pair["nativeVariable"]=="wind_speed_10m"))
        || (variable=="precipitation" && matches!(pair["windowMinutes"].as_u64(),Some(60|360)) && metric=="ets")
        || (variable=="reflectivity" && metric=="fss")
}
/// The headline metrics of the per-lead and per-day tables. Request metadata, never per-template code.
fn default_headline() -> Value {
    json!([
        {"variable":"t2","metric":"rmse"},{"variable":"t2","metric":"bias"},
        {"variable":"td2","metric":"rmse"},{"variable":"td2","metric":"bias"},
        {"variable":"wind10","metric":"rmse"},{"variable":"wind10","metric":"bias"},
        {"variable":"precipitation","metric":"ets","window_minutes":60,"threshold":1.0},
        {"variable":"precipitation","metric":"ets","window_minutes":360,"threshold":5.0},
        {"variable":"reflectivity","metric":"fss","threshold":30.0,"scale_km":25.0}
    ])
}
fn headline(rules: &Value, pair: &Value) -> bool {
    rules.as_array().is_some_and(|rules| rules.iter().any(|r| {
        r["variable"]==pair["variable"] && r["metric"]==pair["metric"]
            && (r["window_minutes"].is_null() || r["window_minutes"].as_u64()==pair["windowMinutes"].as_u64())
            && (r["threshold"].is_null() || r["threshold"].as_f64()==pair["threshold"].as_f64())
            && (r["scale_km"].is_null() || r["scale_km"].as_f64()==pair["scaleKm"].as_f64())
    }))
}
/// One pooled table row's description, from the first pair that opened it.
fn describe(pair: &Value) -> Value {
    json!({"variable":pair["variable"],"metric":pair["metric"],"units":pair["units"],"threshold":pair["threshold"],"scale_km":pair["scaleKm"],"window_minutes":pair["windowMinutes"],"neighborhood_height_km":pair["windowHeightKm"],"neighborhood_width_min_km":pair["windowWidthMinKm"],"neighborhood_width_max_km":pair["windowWidthMaxKm"]})
}
#[derive(Default)]
struct Pool { pair: Value, woof: Stats, comparator: Stats, runs: BTreeSet<String>, leads: BTreeSet<u64> }
impl Pool {
    fn add(&mut self, pair: &Value, w: &Stats, b: &Stats, run: &str, lead: u64) {
        if self.pair.is_null() { self.pair = pair.clone(); }
        self.woof.merge(w); self.comparator.merge(b); self.runs.insert(run.to_owned()); self.leads.insert(lead);
    }
    fn row(&self, with_stats: bool) -> Value {
        let metric=self.pair["metric"].as_str().unwrap_or("");
        let mut row=describe(&self.pair);
        row["n"]=json!(self.woof.count);
        row["forecast"]=json!(self.woof.metric(metric));
        row["comparator"]=json!(self.comparator.metric(metric));
        row["run_count"]=json!(self.runs.len());
        row["lead_count"]=json!(self.leads.len());
        if with_stats { row["forecast_stats"]=json!(self.woof); row["comparator_stats"]=json!(self.comparator); }
        row
    }
}
/// Headline series: one row per owner (template and domain, or a run's domain) and metric, with parallel
/// arrays over the axis (lead seconds or initialization day). Compact enough to keep every run's own.
type Series<A> = BTreeMap<(Vec<String>,String),BTreeMap<A,Pool>>;
fn series<A: Ord + Serialize>(groups: &Series<A>, owner: &[&str], axis: &str) -> Vec<Value> {
    groups.iter().filter_map(|((names,_),points)| {
        let first=points.values().next()?;
        let metric=first.pair["metric"].as_str().unwrap_or("");
        let mut row=describe(&first.pair);
        for (k,v) in owner.iter().zip(names) { row[*k]=json!(v); }
        row[axis]=json!(points.keys().collect::<Vec<_>>());
        row["n"]=json!(points.values().map(|p|p.woof.count).collect::<Vec<_>>());
        row["forecast"]=json!(points.values().map(|p|p.woof.metric(metric)).collect::<Vec<_>>());
        row["comparator"]=json!(points.values().map(|p|p.comparator.metric(metric)).collect::<Vec<_>>());
        row["run_count"]=json!(points.values().map(|p|p.runs.len()).collect::<Vec<_>>());
        Some(row)
    }).collect()
}
fn place(row: Value, fields: Value) -> Value {
    let mut out=fields;
    for (k,v) in row.as_object().into_iter().flatten() { out[k]=v.clone(); }
    out
}

/// Score one frame request, or reuse its verified receipt. `retired`: the inputs were deleted after the
/// frame scored final; only the receipt's own hash can be checked, and a missing receipt is a named gap.
fn score_frame(out: &Path, cache_name: &str, request: &Value, comparator: &str, frame: &Value, init: i64, valid: i64, domain: &str, retired: bool) -> Result<(Value,bool)> {
    let cache=out.join("runs").join(cache_name);
    let prior=fs::read(&cache).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .filter(|p| serde_json::to_vec(&p["score"]).is_ok_and(|s| p["score_hash"]==digest(&s)));
    if retired {
        return prior.map(|p| (p["score"].clone(),true)).ok_or_else(|| fail("inputs_retired_without_verified_receipt"));
    }
    let mut score_request=request.clone();
    if score_request["baseline"].as_str().is_none() {return Err(fail("baseline_extract_missing"));}
    if frame["baseline_model"]!=comparator {return Err(fail("comparator_model_mismatch"));}
    score_request.as_object_mut().ok_or_else(||fail("score_request_missing"))?.remove("output");
    // Hash actual immutable bytes, including QC and station support, before reuse.
    let mut hashes=vec![];
    for field in ["forecast","observations","baseline"] {
        hashes.push(digest(&fs::read(get_str(&score_request,field)?)?));
    }
    let input_hash=digest(&serde_json::to_vec(&json!([SOURCE_REV,METHOD,score_request,hashes]))?);
    if let Some(p)=prior.as_ref() {
        if p["input_hash"]==input_hash {return Ok((p["score"].clone(),true));}
    }
    let value=score::run(&score_request)?;
    if time(get_str(&value,"validTime")?)?!=valid || value["domain"]!=domain || value["baseline_cycle_gap_seconds"]!=0 {
        return Err(fail("same_cycle_and_hour_comparison_missing"));
    }
    if value["leadSeconds"].as_u64()!=Some((valid-init) as u64) {return Err(fail("forecast_initialization_mismatch"));}
    write(&cache,&json!({"input_hash":input_hash,"score_hash":digest(&serde_json::to_vec(&value)?),"score":value}))?;
    Ok((value,false))
}

pub fn run(request: &Value) -> Result<Value> {
    let as_of=time(get_str(request,"as_of")?)?;
    let catalog:Value=serde_json::from_slice(&fs::read(get_str(request,"catalog")?)?)?;
    if catalog["schema"]!=SCHEMA {return Err(fail(format!("expected {SCHEMA}")));}
    // The snapshot must cover the aggregation window, including deleted/private rows.
    let start=time(get_str(&catalog,"window_start")?)?;
    let end=time(get_str(&catalog,"window_end")?)?;
    if start>as_of-30*86400 || end<as_of || !catalog["complete"].as_bool().unwrap_or(false) {
        return Err(fail("catalog does not establish every public run in the rolling 30-day window"));
    }
    let comparator=get_str(&catalog,"comparator")?;
    let rules=if request["headline"].is_array() {request["headline"].clone()} else {default_headline()};
    let out=PathBuf::from(get_str(request,"output_dir")?);
    fs::create_dir_all(out.join("runs"))?;
    fs::create_dir_all(out.join("summaries"))?;
    let lock=OpenOptions::new().create(true).truncate(false).read(true).write(true).open(out.join(".nightly.lock"))?;
    lock.try_lock().map_err(|e|fail(format!("another nightly writer holds the output directory: {e}")))?;
    let mut runs=catalog["runs"].as_array().ok_or_else(||fail("catalog runs missing"))?.clone();
    runs.sort_by(|a,b|a["run_id"].as_str().cmp(&b["run_id"].as_str()));
    let mut ids=BTreeSet::new();
    let mut rows=vec![];
    // (template, domain, key) over the window; headline series per lead and per initialization day.
    let mut templates=BTreeMap::<(String,String,String),Pool>::new();
    let mut leads=Series::<u64>::new();
    let mut days=Series::<String>::new();
    let(mut total7,mut scored7,mut total30,mut scored30,mut reused)=(0u64,0u64,0u64,0u64,0u64);
    for run in runs {
        let id=get_str(&run,"run_id")?;
        if !safe_id(id) || !ids.insert(id.to_owned()) {return Err(fail("unsafe or duplicate run id would overwrite a run receipt"));}
        if !run["is_public"].as_bool().unwrap_or(false) || !run["deleted_at"].is_null() {continue;}
        let init=time(get_str(&run,"init_time")?)?;
        if init<as_of-30*86400 || init>=as_of {continue;}
        let template=get_str(&run,"template")?.to_owned();
        let day=iso(init)[..10].to_owned();
        let in7=init>=as_of-7*86400;
        total30+=1;
        if in7 {total7+=1;}
        let mut row=json!({"run_id":id,"template":template,"init_time":iso(init),"run_status":run["status"],"status":"skipped","reasons":[]});
        let mut reasons=vec![];
        let mut receipts=vec![];
        let frames=run["frames"].as_array().cloned().unwrap_or_default();
        // Every expected frame counts, a skipped one too, so its gap stays visible.
        let frames_total=frames.len();
        if run["status"]!="finished" {reasons.push(json!("run_not_finished"));}
        else if run["observations_final_after"].as_str().is_none() {reasons.push(json!("observation_finality_watermark_missing"));}
        else if time(get_str(&run,"observations_final_after")?)?>as_of {reasons.push(json!("observations_not_final"));}
        else {
            if frames.is_empty(){reasons.push(json!("no_retained_forecast_extracts"));}
            let mut frames=frames;
            frames.sort_by(|a,b|(a["valid_time"].as_str(),a["domain"].as_str()).cmp(&(b["valid_time"].as_str(),b["domain"].as_str())));
            let mut frame_ids=BTreeSet::new();
            for frame in frames {
                let valid=time(get_str(&frame,"valid_time")?)?;
                let domain=get_str(&frame,"domain")?;
                if valid<init || valid>as_of || !safe_id(domain) || !frame_ids.insert((valid,domain.to_owned())) {
                    return Err(fail("invalid or duplicate frame would double count the same forecast hour"));
                }
                // The driver's own named gap (no comparator at this lead, no observation coverage, an expired
                // history file): recorded as the frame's reason, never scored.
                if let Some(reason)=frame["skip_reason"].as_str() {
                    reasons.push(json!({"valid_time":iso(valid),"domain":domain,"reason":reason}));
                    continue;
                }
                // One frame may need several score requests (the 6-hour rain window has its own forecast
                // artifact); each keeps its own receipt so a frame is never counted twice.
                let requests:Vec<Value>=match frame["score_requests"].as_array() {
                    Some(list)=>list.clone(),
                    None=>vec![frame["score_request"].clone()],
                };
                let retired=frame["inputs_retired"].as_bool()==Some(true);
                for (k,score_request) in requests.iter().enumerate() {
                    let name=if k==0 {format!("{id}-{domain}-{valid}.json")} else {format!("{id}-{domain}-{valid}-{k}.json")};
                    match score_frame(&out,&name,score_request,comparator,&frame,init,valid,domain,retired) {
                        Ok((value,cached))=>{reused+=u64::from(cached);receipts.push(value);}
                        Err(error)=>reasons.push(json!({"valid_time":iso(valid),"domain":domain,"reason":error.to_string()})),
                    }
                }
            }
        }
        let mut metric_ids=BTreeSet::new();
        let mut run_pool=BTreeMap::<(String,String),Pool>::new();
        let mut run_leads=Series::<u64>::new();
        let mut scored_frames=BTreeSet::new();
        let mut pooled_once=BTreeSet::new();
        for receipt in &receipts {
            let domain=get_str(receipt,"domain")?.to_owned();
            let lead=receipt["leadSeconds"].as_u64().ok_or_else(||fail("score lead missing"))?;
            for pair in receipt["pairs"].as_array().ok_or_else(||fail("native score pairs missing"))? {
                if !wanted(pair){continue;}
                let w:Stats=serde_json::from_value(pair["woof"].clone())?;
                let b:Stats=serde_json::from_value(pair["baseline"].clone())?;
                if w.count!=b.count || w.count==0 {return Err(fail("paired score counts differ"));}
                let k=key(pair)?;
                // Two requests of one frame scoring the same metric would count that hour twice.
                if !pooled_once.insert((domain.clone(),lead,k.clone())) {
                    let reason=json!({"valid_time":get_str(receipt,"validTime")?,"domain":domain,"reason":"overlapping_score_requests"});
                    if !reasons.contains(&reason) {reasons.push(reason);}
                    continue;
                }
                metric_ids.insert(pair["variable"].as_str().unwrap_or("").to_owned());
                scored_frames.insert((domain.clone(),lead));
                templates.entry((template.clone(),domain.clone(),k.clone())).or_default().add(pair,&w,&b,id,lead);
                run_pool.entry((domain.clone(),k.clone())).or_default().add(pair,&w,&b,id,lead);
                if headline(&rules,pair) {
                    let owner=vec![template.clone(),domain.clone()];
                    leads.entry((owner.clone(),k.clone())).or_default().entry(lead).or_default().add(pair,&w,&b,id,lead);
                    days.entry((owner,k.clone())).or_default().entry(day.clone()).or_default().add(pair,&w,&b,id,lead);
                    run_leads.entry((vec![domain.clone()],k)).or_default().entry(lead).or_default().add(pair,&w,&b,id,lead);
                }
            }
        }
        for variable in ["t2","td2","wind10","precipitation","reflectivity"] {
            if !receipts.is_empty() && !metric_ids.contains(variable){reasons.push(json!(format!("metric_unavailable:{variable}")));}
        }
        let mut summary_sha256=Value::Null;
        if !metric_ids.is_empty() {
            row["status"]=json!(if reasons.is_empty(){"scored"}else{"partial"});
            if in7 {scored7+=1;}
            scored30+=1;
            let summary=json!({"schema":SUMMARY_SCHEMA,"run_id":id,"template":template,"init_time":iso(init),"method_id":METHOD,"comparator":comparator,
                "status":row["status"],"reasons":reasons,"frames_scored":scored_frames.len(),"frames_total":frames_total,
                "scores":run_pool.iter().map(|((domain,_),p)| place(p.row(true),json!({"domain":domain}))).collect::<Vec<_>>(),
                "lead_series":series(&run_leads,&["domain"],"lead_seconds")});
            summary_sha256=json!(write(&out.join("summaries").join(format!("{id}.json")),&summary)?);
        } else if !receipts.is_empty() {
            reasons.push(json!("no_paired_metric"));
        }
        row["reasons"]=json!(reasons);
        row["frames_scored"]=json!(scored_frames.len());
        row["frames_total"]=json!(frames_total);
        row["summary_sha256"]=summary_sha256;
        rows.push(row);
    }
    let table:Vec<_>=templates.iter().map(|((template,domain,_),p)| place(p.row(true),json!({"template":template,"domain":domain}))).collect();
    let lead_table=series(&leads,&["template","domain"],"lead_seconds");
    let day_table=series(&days,&["template","domain"],"days");
    let result=json!({"schema":"woof.scoreboard.nightly.v2","as_of":iso(as_of),"method_id":METHOD,"comparator":comparator,"window_start":iso(as_of-30*86400),
        "seven_day":{"scored":scored7,"total":total7},"thirty_day":{"scored":scored30,"total":total30},"headline":rules,
        "runs":rows,"templates":table,"template_lead_series":lead_table,"template_day_series":day_table});
    write(&out.join("nightly.json"),&result)?;
    // Reuse counts are process telemetry, excluded from the stable publication.
    eprintln!("nightly frames reused: {reused}");
    Ok(result)
}
