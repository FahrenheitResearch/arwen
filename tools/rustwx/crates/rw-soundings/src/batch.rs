use crate::{
    analysis, api,
    input::{self, Frame, Point, Source},
    options,
    render::{self, Job, MemberTrace, Overlay, RenderRequest, SoundingScene},
    store::{self, Res},
};
use rayon::prelude::*;
use rustwx_sounding::SoundingMetadata;
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};
#[derive(Deserialize)]
pub struct Request {
    pub sources: Vec<Source>,
    pub points: Vec<Point>,
    pub out: PathBuf,
    #[serde(default = "default_modes")]
    pub modes: Vec<String>,
    #[serde(default = "default_interval")]
    pub interval_minutes: u32,
    #[serde(default)]
    pub settings: options::Q,
    #[serde(default = "default_threads")]
    pub threads: usize,
}
fn default_modes() -> Vec<String> {
    vec!["mean".into(), "all".into()]
}
fn default_interval() -> u32 {
    60
}
fn default_threads() -> usize {
    4
}
fn stamp(t: i64, fmt: &str) -> String {
    chrono::DateTime::from_timestamp(t, 0)
        .unwrap()
        .format(fmt)
        .to_string()
}
fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
pub fn run(req: Request) -> Res<()> {
    if req.sources.is_empty() || req.points.is_empty() {
        return Err("a sounding needs input files and at least one point".into());
    }
    if req.interval_minutes == 0 {
        return Err("interval_minutes must be positive".into());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(req.threads.max(1))
        .build()
        .map_err(|e| e.to_string())?;
    let started = std::time::Instant::now();
    let frames: Vec<Frame> = pool
        .install(|| {
            req.sources
                .par_iter()
                .map(|s| input::read(s, &req.points))
                .collect::<Res<Vec<_>>>()
        })?
        .into_iter()
        .flatten()
        .collect();
    let mut groups: BTreeMap<(String, i64), BTreeMap<String, Frame>> = BTreeMap::new();
    let mut origins: BTreeMap<String, i64> = BTreeMap::new();
    let mut roster: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    for f in frames {
        origins
            .entry(f.domain.clone())
            .and_modify(|t| *t = (*t).min(f.valid))
            .or_insert(f.valid);
        roster
            .entry(f.domain.clone())
            .or_default()
            .insert(f.member.clone());
        let key = (f.domain.clone(), f.valid);
        let member = f.member.clone();
        if groups
            .entry(key)
            .or_default()
            .insert(member.clone(), f)
            .is_some()
        {
            return Err(format!(
                "duplicate input for member {member} at the same valid time; select one history/store inventory"
            ));
        }
    }
    eprintln!(
        "soundings: read {} time groups in {:.2}s using {} CPU workers",
        groups.len(),
        started.elapsed().as_secs_f64(),
        req.threads
    );
    let tx = render::spawn_pool(2)?;
    let ao = options::parse_analysis(&req.settings)?;
    let display = options::parse_display(&req.settings)?;
    let mut pages: BTreeMap<PathBuf, Vec<(String, String)>> = BTreeMap::new();
    let mut receipt = vec![];
    let mut pending = std::collections::VecDeque::new();
    for ((domain, valid), frames) in groups {
        if (valid - origins[&domain]) % (i64::from(req.interval_minutes) * 60) != 0 {
            continue;
        }
        if frames
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            != roster[&domain]
        {
            return Err(format!(
                "{domain} {} has missing members; an ensemble band requires matching valid times",
                stamp(valid, "%FT%TZ")
            ));
        }
        let folder = req
            .out
            .join(slug(&domain))
            .join("soundings")
            .join(stamp(valid, "%Y-%m-%d"));
        std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        let names: Vec<String> = frames.keys().cloned().collect();
        for (pi, point) in req.points.iter().enumerate() {
            let raw: Vec<_> = frames.values().map(|f| f.columns[pi].clone()).collect();
            let common = input::common_columns(&raw);
            let meta = SoundingMetadata {
                station_id: point.label.clone(),
                valid_time: stamp(valid, "%Y-%m-%d %H:%MZ"),
                latitude_deg: Some(point.lat),
                longitude_deg: Some(point.lon),
                sample_method: Some(
                    "bilinear; common log-pressure levels for ensemble statistics".into(),
                ),
                ..Default::default()
            };
            let analyze = |r: &store::RawColumn| {
                analysis::analyze(analysis::to_column(r, meta.clone())?, &ao)
            };
            let members: Vec<_> =
                pool.install(|| raw.par_iter().map(analyze).collect::<Res<Vec<_>>>())?;
            for mode in &req.modes {
                let title = format!("{}  {}  {}  {}", domain, point.label, meta.valid_time, mode);
                let main = if mode == "mean" || mode == "all" {
                    analyze(&store::mean_columns(&common)?)?
                } else {
                    let i = names.iter().position(|n| n == mode).ok_or_else(|| {
                        format!("unknown member {mode}; available: {}", names.join(", "))
                    })?;
                    analyze(&raw[i])?
                };
                let overlay = if mode == "all" {
                    let refs = members.iter().collect::<Vec<_>>();
                    Some(Overlay {
                        members: members
                            .iter()
                            .zip(&names)
                            .map(|(a, n)| MemberTrace {
                                label: n.clone(),
                                p: a.column.pressure_hpa.clone(),
                                t: a.column.temperature_c.clone(),
                                td: a.column.dewpoint_c.clone(),
                                hodo: api::hodo_trace(a),
                            })
                            .collect(),
                        bands: Some(api::bands_from(&common, &refs)),
                        inner_band: false,
                        show_t: display.spag_t,
                        show_td: display.spag_td,
                        alpha: display.spag_alpha,
                        legend: format!(
                            "{} members (thin) · P10-P90 band · thick = ensemble mean",
                            members.len()
                        ),
                    })
                } else {
                    None
                };
                let stem = format!(
                    "{}_{}_{}",
                    slug(&point.label),
                    stamp(valid, "%Y%m%dT%H%M%SZ"),
                    slug(mode)
                );
                let blank: Vec<_> = display.blank.iter().map(|(_, id)| id.clone()).collect();
                let mut data = json!({"schema":"gpuwm-sounding.v1","title":title,"domain":domain,"valid":stamp(valid,"%FT%TZ"),"point":{"label":point.label,"lat":point.lat,"lon":point.lon},"mode":mode,"members_used":names,"column":main.column,"analysis":analysis::analysis_json(&main),"tables":crate::tables::tables_json(&main,&blank)});
                if mode == "all" {
                    data["percentiles"] = store::percentiles5(&common);
                    data["members"]=json!(members.iter().zip(&names).map(|(a,n)|json!({"member":n,"column":a.column,"indices":analysis::summary_json(a),"hodograph_kt":api::hodo_trace(a)})).collect::<Vec<_>>());
                }
                let scene = SoundingScene {
                    patches: api::patches(&display, &main),
                    analysis: Arc::new(main),
                    view: display.view.clone(),
                    layout: display.layout.clone(),
                    corner: display.corner,
                    parcel: display.parcel,
                    title: title.clone(),
                    brand: display.brand.clone(),
                    style: display.style.clone(),
                    overlay,
                };
                let (reply, rx) = tokio::sync::oneshot::channel();
                tx.send(RenderRequest {
                    job: Arc::new(Job::Sounding(scene)),
                    size: egui::Vec2::new(display.width, display.height),
                    ppp: display.ppp,
                    reply,
                })
                .map_err(|e| e.to_string())?;
                pending.push_back((folder.clone(), stem, title, data, rx));
                if pending.len() >= 2 {
                    finish(pending.pop_front().unwrap(), &mut pages, &mut receipt)?;
                }
            }
        }
    }
    while let Some(job) = pending.pop_front() {
        finish(job, &mut pages, &mut receipt)?;
    }
    if receipt.is_empty() {
        return Err("no requested sounding times were available".into());
    }
    for (folder, rows) in pages {
        let mut html = String::from(
            "<!doctype html><meta charset=utf-8><title>Soundings</title><style>body{background:#101018;color:#eee;font:16px sans-serif}img{max-width:100%;}a{color:#8cf}figure{margin:12px 0}</style><h1>Soundings</h1><p>Hourly profiles; mean, members and ensemble P10–P90 spread.</p>",
        );
        for (stem, title) in rows {
            html.push_str(&format!("<figure><figcaption>{} · <a href=\"{}.json\">profile and indices JSON</a></figcaption><a href=\"{}.png\"><img loading=lazy src=\"{}.png\" alt=\"{}\"></a></figure>",escape(&title),stem,stem,stem,escape(&title)));
        }
        std::fs::write(folder.join("index.html"), html).map_err(|e| e.to_string())?;
    }
    std::fs::write(req.out.join("soundings-manifest.json"),serde_json::to_vec_pretty(&json!({"schema":"gpuwm-soundings.v1","pictures":receipt,"wall_seconds":started.elapsed().as_secs_f64()})).unwrap()).map_err(|e|e.to_string())?;
    Ok(())
}

// Two queued windows overlap the independent egui frame/PNG stages without retaining a time series in RAM.
type Pending = (
    PathBuf,
    String,
    String,
    serde_json::Value,
    tokio::sync::oneshot::Receiver<Res<render::Rendered>>,
);
fn finish(
    (folder, stem, title, data, rx): Pending,
    pages: &mut BTreeMap<PathBuf, Vec<(String, String)>>,
    receipt: &mut Vec<serde_json::Value>,
) -> Res<()> {
    let rendered = rx.blocking_recv().map_err(|e| e.to_string())??;
    std::fs::write(folder.join(format!("{stem}.png")), rendered.png).map_err(|e| e.to_string())?;
    std::fs::write(
        folder.join(format!("{stem}.json")),
        serde_json::to_vec_pretty(&data).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    pages
        .entry(folder.clone())
        .or_default()
        .push((stem.clone(), title));
    receipt.push(json!({"png":folder.join(format!("{stem}.png")),"frame_ms":rendered.frame_ms,"encode_ms":rendered.encode_ms}));
    println!("RENDERED {}", folder.join(format!("{stem}.png")).display());
    Ok(())
}
