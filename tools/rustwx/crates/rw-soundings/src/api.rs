//! Website sounding API and shared plot helpers:
//!   GET /api/health
//!   GET /api/cycles
//!   GET /api/options
//!   GET /api/sounding.{png,json,txt}   one analysed sounding (point or box)
//!   GET /api/ensemble.{png,json}       mean + every member (mode=spaghetti|envelope)
//!   GET /api/timeheight.{png,json}     time-height across every lead at a point/box

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::Router;
use axum::extract::{Path as AxPath, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use egui::{Color32, Vec2};
use lru::LruCache;
use rayon::prelude::*;
use rustwx_sounding::SoundingMetadata;
use serde_json::{Value, json};

use crate::analysis::{self, Analysis, analysis_json};
use crate::options::{self, DisplayOpts, Q, qf, qs};
use crate::render::{
    self, Bands, Job, MemberTrace, Overlay, Ramp, RenderRequest, SoundingScene, TimeHeightScene,
};
use crate::store::{self, RawColumn, Store};
use crate::tables;

pub struct AppState {
    pub store: Store,
    pub render_tx: Mutex<Sender<RenderRequest>>,
    pub cache: Mutex<ByteCache>,
    pub cells: Mutex<LruCache<String, Arc<Vec<(usize, usize)>>>>,
    pub started: Instant,
}

type Cached = (Arc<Vec<u8>>, &'static str, Vec<(String, String)>);

pub struct ByteCache {
    map: LruCache<String, Cached>,
    bytes: usize,
    budget: usize,
}

impl ByteCache {
    pub fn new(budget: usize) -> Self {
        ByteCache {
            map: LruCache::new(NonZeroUsize::new(100_000).unwrap()),
            bytes: 0,
            budget,
        }
    }
    fn get(&mut self, k: &str) -> Option<Cached> {
        self.map.get(k).cloned()
    }
    fn put(&mut self, k: String, v: Cached) {
        self.bytes += v.0.len();
        if let Some(old) = self.map.put(k, v) {
            self.bytes -= old.0.len();
        }
        while self.bytes > self.budget {
            match self.map.pop_lru() {
                Some((_, old)) => self.bytes -= old.0.len(),
                None => break,
            }
        }
    }
}

/// Typed API error -> status + JSON body per the contract.
pub enum ApiErr {
    Bad(String),
    NotFound(String, Value),
    OffGrid(String, Value),
    Internal(String),
}

impl From<String> for ApiErr {
    fn from(s: String) -> Self {
        ApiErr::Bad(s)
    }
}
impl From<&str> for ApiErr {
    fn from(s: &str) -> Self {
        ApiErr::Bad(s.to_string())
    }
}

type R<T> = Result<T, ApiErr>;

fn err_response(e: ApiErr) -> Response {
    let (code, body) = match e {
        ApiErr::Bad(m) => (StatusCode::BAD_REQUEST, json!({"error": m})),
        ApiErr::NotFound(m, have) => (StatusCode::NOT_FOUND, json!({"error": m, "have": have})),
        ApiErr::OffGrid(m, grid) => (StatusCode::BAD_REQUEST, json!({"error": m, "grid": grid})),
        ApiErr::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, json!({"error": m})),
    };
    (
        code,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/{name}", get(dispatch))
        .route("/api/sounding/{name}", get(legacy_dispatch))
        .with_state(state)
}

fn fnv(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("\"{h:016x}\"")
}

async fn dispatch(
    State(st): State<Arc<AppState>>,
    AxPath(name): AxPath<String>,
    Query(q): Query<Q>,
    headers: HeaderMap,
) -> Response {
    let t0 = Instant::now();
    let (base, fmt) = match name.rsplit_once('.') {
        Some((b, f)) => (b.to_string(), f.to_string()),
        None => (name.clone(), String::new()),
    };
    let kind = match (base.as_str(), fmt.as_str()) {
        ("health", _) => return health(&st).await,
        ("cycles", _) => return blocking_json(st.clone(), cycles).await,
        ("options", _) => return json_resp(options::schema(), "no-cache"),
        ("sounding", "png" | "json" | "txt") => Kind::Sounding,
        ("ensemble", "png" | "json") => Kind::Ensemble,
        ("timeheight", "png" | "json") => Kind::TimeHeight,
        _ => {
            return err_response(ApiErr::NotFound(
                format!("unknown endpoint '/api/{name}'"),
                json!([
                    "health",
                    "cycles",
                    "options",
                    "sounding.png",
                    "sounding.json",
                    "sounding.txt",
                    "ensemble.png",
                    "ensemble.json",
                    "timeheight.png",
                    "timeheight.json"
                ]),
            ));
        }
    };
    finish(st, kind, fmt, q, headers, t0).await
}

async fn finish(
    st: Arc<AppState>,
    kind: Kind,
    fmt: String,
    q: Q,
    headers: HeaderMap,
    t0: Instant,
) -> Response {
    let st2 = st.clone();
    let fmt2 = fmt.clone();
    let prep = tokio::task::spawn_blocking(move || prepare(&st2, kind, &fmt2, &q)).await;
    let prep = match prep {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => return err_response(e),
        Err(e) => return err_response(ApiErr::Internal(e.to_string())),
    };
    let inm = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    match prep {
        Prepared::Cached(key, (bytes, ctype, hdrs)) => {
            respond(&key, bytes, ctype, hdrs, inm, t0, true)
        }
        Prepared::Bytes(key, bytes, ctype, hdrs) => {
            let bytes = Arc::new(bytes);
            st.cache
                .lock()
                .unwrap()
                .put(key.clone(), (bytes.clone(), ctype, hdrs.clone()));
            respond(&key, bytes, ctype, hdrs, inm, t0, false)
        }
        Prepared::Render(key, job, size, ppp, mut hdrs, prep_ms) => {
            let (tx, rx) = tokio::sync::oneshot::channel();
            if st
                .render_tx
                .lock()
                .unwrap()
                .send(RenderRequest {
                    job,
                    size,
                    ppp,
                    reply: tx,
                })
                .is_err()
            {
                return err_response(ApiErr::Internal("render pool is down".into()));
            }
            match rx.await {
                Ok(Ok(r)) => {
                    hdrs.push((
                        "x-sounding-ms".into(),
                        format!("{:.1}", prep_ms + r.frame_ms),
                    ));
                    let timing = hdrs
                        .iter()
                        .find(|h| h.0 == "x-sounding-timing")
                        .map(|h| h.1.clone())
                        .unwrap_or_default();
                    hdrs.retain(|h| h.0 != "x-sounding-timing");
                    hdrs.push((
                        "x-sounding-timing".into(),
                        format!(
                            "{timing}; frame={:.1}ms; png={:.1}ms; px={}x{}",
                            r.frame_ms, r.encode_ms, r.width, r.height
                        ),
                    ));
                    let bytes = Arc::new(r.png);
                    st.cache
                        .lock()
                        .unwrap()
                        .put(key.clone(), (bytes.clone(), "image/png", hdrs.clone()));
                    respond(&key, bytes, "image/png", hdrs, inm, t0, false)
                }
                Ok(Err(e)) => err_response(ApiErr::Internal(format!("render: {e}"))),
                Err(_) => err_response(ApiErr::Internal("render thread dropped the job".into())),
            }
        }
    }
}

/// The first route set (/api/sounding/...), which the site is built on:
/// meta, point.{png,json}, box.{png,json}, series.{png,json}, health.
async fn legacy_dispatch(
    State(st): State<Arc<AppState>>,
    AxPath(name): AxPath<String>,
    Query(mut q): Query<Q>,
    headers: HeaderMap,
) -> Response {
    let t0 = Instant::now();
    let (base, fmt) = match name.rsplit_once('.') {
        Some((b, f)) => (b.to_string(), f.to_string()),
        None => (name.clone(), String::new()),
    };
    let kind = match (base.as_str(), fmt.as_str()) {
        ("health", _) => return health(&st).await,
        ("meta", _) => return blocking_json(st.clone(), legacy_meta).await,
        ("point" | "box", "png") => {
            if base == "box"
                && qs(&q, "bbox").is_none()
                && qs(&q, "box").is_none()
                && qs(&q, "poly").is_none()
            {
                return err_response(ApiErr::Bad(
                    "box needs bbox=w,s,e,n or poly=lon,lat;lon,lat;...".into(),
                ));
            }
            if qs(&q, "member") == Some("all") || qs(&q, "member") == Some("spaghetti") {
                Kind::Ensemble
            } else {
                Kind::Sounding
            }
        }
        ("point" | "box", "json") => Kind::Sounding,
        ("series", "png" | "json") => Kind::TimeHeight,
        _ => {
            return err_response(ApiErr::NotFound(
                format!("unknown endpoint '/api/sounding/{name}'"),
                json!([
                    "health",
                    "meta",
                    "point.png",
                    "point.json",
                    "box.png",
                    "box.json",
                    "series.png",
                    "series.json"
                ]),
            ));
        }
    };
    if base == "point" {
        q.remove("bbox");
        q.remove("box");
        q.remove("poly");
    }
    q.insert("__legacy".into(), "1".into());
    finish(st, kind, fmt, q, headers, t0).await
}

fn legacy_meta(st: &AppState) -> R<Value> {
    let cat = st.store.catalog();
    let mut cycles = vec![];
    for (cycle, members) in cat.iter().rev() {
        let mut leads: BTreeMap<u32, usize> = BTreeMap::new();
        for (m, ls) in members {
            for l in ls.keys() {
                let e = leads.entry(*l).or_default();
                if m != "mean" {
                    *e += 1;
                }
            }
        }
        cycles.push(json!({
            "cycle": cycle,
            "members": members.keys().filter(|m| m.as_str() != "mean").collect::<Vec<_>>(),
            "has_mean": members.contains_key("mean"),
            "leads_min": leads.keys().collect::<Vec<_>>(),
            "members_per_lead": leads,
        }));
    }
    let g = grid_json(st);
    Ok(json!({
        "model": st.store.model,
        "latest": cat.keys().next_back(),
        "cycles": cycles,
        "grid": {"west": g["bbox"]["west"], "south": g["bbox"]["south"], "east": g["bbox"]["east"], "north": g["bbox"]["north"], "nx": g["nx"], "ny": g["ny"], "dx_km": g["dx_km"], "corners_latlon": g["corners_latlon"]},
        "options": options::legacy_catalogue(),
        "endpoints": {
            "point": "/api/sounding/point.{png|json}?lat=&lon=",
            "box": "/api/sounding/box.{png|json}?bbox=w,s,e,n | poly=lon,lat;lon,lat;...",
            "series": "/api/sounding/series.{png|json}?lat=&lon= (or bbox/poly)",
            "meta": "/api/sounding/meta"
        }
    }))
}

fn respond(
    key: &str,
    bytes: Arc<Vec<u8>>,
    ctype: &'static str,
    hdrs: Vec<(String, String)>,
    inm: Option<String>,
    t0: Instant,
    hit: bool,
) -> Response {
    let etag = fnv(key);
    let mut h = HeaderMap::new();
    h.insert(header::ETAG, HeaderValue::from_str(&etag).unwrap());
    h.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=300"),
    );
    for (k, v) in &hdrs {
        if let (Ok(k), Ok(v)) = (
            axum::http::HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(&v.replace(['\n', '\r'], " ")),
        ) {
            h.insert(k, v);
        }
    }
    h.insert(
        "x-sounding-cache",
        HeaderValue::from_static(if hit { "hit" } else { "miss" }),
    );
    h.insert(
        "x-sounding-total-ms",
        HeaderValue::from_str(&format!("{:.2}", t0.elapsed().as_secs_f64() * 1e3)).unwrap(),
    );
    if inm.as_deref() == Some(etag.as_str()) {
        return (StatusCode::NOT_MODIFIED, h).into_response();
    }
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static(ctype));
    (StatusCode::OK, h, (*bytes).clone()).into_response()
}

fn json_resp(v: Value, cache: &'static str) -> Response {
    (
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, cache),
        ],
        v.to_string(),
    )
        .into_response()
}

async fn blocking_json(st: Arc<AppState>, f: fn(&AppState) -> R<Value>) -> Response {
    match tokio::task::spawn_blocking(move || f(&st)).await {
        Ok(Ok(v)) => json_resp(v, "no-cache"),
        Ok(Err(e)) => err_response(e),
        Err(e) => err_response(ApiErr::Internal(e.to_string())),
    }
}

async fn health(st: &Arc<AppState>) -> Response {
    let st2 = st.clone();
    let n = tokio::task::spawn_blocking(move || st2.store.catalog().len())
        .await
        .unwrap_or(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    json_resp(
        json!({
            "ok": true,
            "version": concat!("gpuwm-sounding ", env!("CARGO_PKG_VERSION")),
            "renderer": "sharppyrs (rusty-weather 369d65c + SoundingView omega/interp passthrough)",
            "store": st.store.root.display().to_string(),
            "cycles": n,
            "now": iso(now),
            "uptime_s": st.started.elapsed().as_secs(),
        }),
        "no-cache",
    )
}

/// unix -> "YYYY-MM-DDTHH:MM:SSZ"
fn iso(t: i64) -> String {
    let days = t.div_euclid(86400);
    let secs = t.rem_euclid(86400);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        secs / 3600,
        (secs % 3600) / 60,
        secs % 60
    )
}

fn cycle_unix(cycle: &str) -> i64 {
    let n = |a: usize, b: usize| {
        cycle
            .get(a..b)
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0)
    };
    let (y, m, d, hh, mm) = (n(0, 4), n(4, 6), n(6, 8), n(8, 10), n(10, 12));
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097 + doe - 719468) * 86400 + hh * 3600 + mm * 60
}

fn valid_iso(cycle: &str, lead: u32) -> String {
    iso(cycle_unix(cycle) + lead as i64 * 60)
}

fn grid_json(st: &AppState) -> Value {
    let cat = st.store.catalog();
    let Some(path) = cat
        .values()
        .next_back()
        .and_then(|m| m.values().next())
        .and_then(|l| l.values().next())
    else {
        return Value::Null;
    };
    let Ok(g) = st.store.grid_for(path) else {
        return Value::Null;
    };
    let (nx, ny) = (g.grid.nx, g.grid.ny);
    let at = |ix: usize, iy: usize| {
        let i = iy * nx + ix;
        json!([g.grid.lat[i], g.grid.lon[i]])
    };
    // row 0 may be the south or the north edge; report by latitude
    let (s_row, n_row) = if g.grid.lat[0] < g.grid.lat[(ny - 1) * nx] {
        (0, ny - 1)
    } else {
        (ny - 1, 0)
    };
    let dx = {
        let (a, b) = ((ny / 2) * nx + nx / 2, (ny / 2) * nx + nx / 2 + 1);
        let (la1, lo1, la2, lo2) = (
            g.grid.lat[a] as f64,
            g.grid.lon[a] as f64,
            g.grid.lat[b] as f64,
            g.grid.lon[b] as f64,
        );
        let x = (lo2 - lo1).to_radians() * ((la1 + la2) / 2.0).to_radians().cos();
        let y = (la2 - la1).to_radians();
        6371.0 * x.hypot(y)
    };
    json!({
        "corners_latlon": {"sw": at(0, s_row), "se": at(nx - 1, s_row), "nw": at(0, n_row), "ne": at(nx - 1, n_row)},
        "nx": nx, "ny": ny, "dx_km": (dx * 100.0).round() / 100.0,
        "bbox": {"west": g.bbox[0], "south": g.bbox[1], "east": g.bbox[2], "north": g.bbox[3]},
    })
}

fn cycles(st: &AppState) -> R<Value> {
    let cat = st.store.catalog();
    let mut out = vec![];
    for (cycle, members) in cat.iter().rev() {
        let mut leads: BTreeMap<u32, (Vec<String>, bool)> = BTreeMap::new();
        for (m, ls) in members {
            for l in ls.keys() {
                let e = leads.entry(*l).or_default();
                if m == "mean" {
                    e.1 = true;
                } else {
                    e.0.push(m.clone());
                }
            }
        }
        out.push(json!({
            "cycle": cycle,
            "valid": valid_iso(cycle, 0),
            "leads": leads.iter().map(|(l, (ms, mean))| json!({
                "lead": l, "valid": valid_iso(cycle, *l), "kind": if *l == 0 { "analysis" } else { "forecast" },
                "members": ms, "mean": *mean || ms.len() > 1,
            })).collect::<Vec<_>>(),
        }));
    }
    Ok(json!({"cycles": out, "grid": grid_json(st), "model": st.store.model}))
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Sounding,
    Ensemble,
    TimeHeight,
}

enum Prepared {
    Cached(String, Cached),
    Bytes(String, Vec<u8>, &'static str, Vec<(String, String)>),
    Render(String, Arc<Job>, Vec2, f32, Vec<(String, String)>, f64),
}

enum Where {
    Point {
        lat: f64,
        lon: f64,
    },
    Area {
        poly: Vec<(f64, f64)>,
        bbox: Option<[f64; 4]>,
        label: String,
    },
}

fn parse_where(q: &Q) -> R<Where> {
    let bbox = qs(q, "box").or_else(|| qs(q, "bbox"));
    if let Some(b) = bbox {
        let v: Vec<f64> = b.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if v.len() != 4 || v[0] >= v[2] || v[1] >= v[3] {
            return Err("box must be w,s,e,n with w<e and s<n".into());
        }
        let ns = |x: f64| if x < 0.0 { "S" } else { "N" };
        let ew = |x: f64| if x < 0.0 { "W" } else { "E" };
        return Ok(Where::Area {
            poly: vec![(v[0], v[1]), (v[2], v[1]), (v[2], v[3]), (v[0], v[3])],
            bbox: Some([v[0], v[1], v[2], v[3]]),
            label: format!(
                "{:.2}..{:.2}{} {:.2}..{:.2}{}",
                v[1].abs(),
                v[3].abs(),
                ns(v[3]),
                v[0].abs(),
                v[2].abs(),
                ew(v[2])
            ),
        });
    }
    if let Some(p) = qs(q, "poly") {
        let pts: Vec<(f64, f64)> = p
            .split(';')
            .filter_map(|pair| {
                let mut it = pair.split(',').map(|x| x.trim().parse::<f64>());
                match (it.next(), it.next()) {
                    (Some(Ok(lon)), Some(Ok(lat))) => Some((lon, lat)),
                    _ => None,
                }
            })
            .collect();
        if pts.len() < 3 {
            return Err("poly needs at least 3 lon,lat vertices separated by ';'".into());
        }
        let n = pts.len();
        return Ok(Where::Area {
            poly: pts,
            bbox: None,
            label: format!("polygon ({n} vertices)"),
        });
    }
    let lat = qf(q, "lat")?.ok_or("lat and lon (or box=w,s,e,n) are required")?;
    let lon = qf(q, "lon")?.ok_or("lat and lon (or box=w,s,e,n) are required")?;
    Ok(Where::Point { lat, lon })
}

/// Canonical cache key: endpoint + resolved selection + every param sorted.
fn cache_key(kind: &str, cycle: &str, lead: &str, q: &Q) -> String {
    let mut items: Vec<(&String, &String)> = q
        .iter()
        .filter(|(k, _)| k.as_str() != "cycle" && k.as_str() != "lead")
        .collect();
    items.sort();
    let rest: Vec<String> = items.iter().map(|(k, v)| format!("{k}={v}")).collect();
    format!("{kind}|{cycle}|{lead}|{}", rest.join("&"))
}

struct Columns {
    members: Vec<String>,
    cols: Vec<RawColumn>,
    lat: f64,
    lon: f64,
    fx: Option<f64>,
    fy: Option<f64>,
    footprint: Option<(f64, f64, f64, f64)>,
    cells: usize,
    label: String,
    read_ms: f64,
}

fn read_columns(st: &AppState, paths: &[(String, PathBuf)], wh: &Where) -> R<Columns> {
    let t = Instant::now();
    let first = &paths.first().ok_or("no members selected")?.1;
    let grid = st.store.grid_for(first)?;
    let grid_v = || json!({"west": grid.bbox[0], "south": grid.bbox[1], "east": grid.bbox[2], "north": grid.bbox[3], "nx": grid.grid.nx, "ny": grid.grid.ny});
    match wh {
        Where::Point { lat, lon } => {
            let off = || {
                ApiErr::OffGrid(
                    format!("point {lat:.3},{lon:.3} is outside the model grid"),
                    grid_v(),
                )
            };
            let (fx, fy) = grid.locator.locate(*lat, *lon).ok_or_else(off)?;
            let (nx, ny) = (grid.grid.nx as f64, grid.grid.ny as f64);
            if fx < 0.0 || fy < 0.0 || fx > nx - 1.0 || fy > ny - 1.0 {
                return Err(off());
            }
            let cols: Vec<Result<RawColumn, String>> = paths
                .par_iter()
                .map(|(_, p)| {
                    let r = st.store.reader(p)?;
                    Store::point_column(&r, fx, fy)
                })
                .collect();
            Ok(Columns {
                members: paths.iter().map(|x| x.0.clone()).collect(),
                cols: cols.into_iter().collect::<Result<Vec<_>, _>>()?,
                lat: *lat,
                lon: *lon,
                fx: Some(fx),
                fy: Some(fy),
                footprint: None,
                cells: 1,
                label: format!(
                    "{:.3}{} {:.3}{}",
                    lat.abs(),
                    if *lat < 0.0 { "S" } else { "N" },
                    lon.abs(),
                    if *lon < 0.0 { "W" } else { "E" }
                ),
                read_ms: t.elapsed().as_secs_f64() * 1e3,
            })
        }
        Where::Area { poly, label, .. } => {
            let key = format!("{}|{:?}", grid.grid.hash, poly);
            let cached = st.cells.lock().unwrap().get(&key).cloned();
            let cells = match cached {
                Some(c) => c,
                None => {
                    let c = Arc::new(store::select_cells(&grid.grid, poly));
                    st.cells.lock().unwrap().put(key, c.clone());
                    c
                }
            };
            if cells.is_empty() {
                return Err(ApiErr::OffGrid(
                    "no model grid cells fall inside the box".into(),
                    grid_v(),
                ));
            }
            if cells.len() > 40_000 {
                return Err(format!("box selects {} cells; limit 40000", cells.len()).into());
            }
            let (mut w, mut s, mut e, mut n) = (
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            );
            for (ix, iy) in cells.iter() {
                let i = iy * grid.grid.nx + ix;
                let (la, lo) = (grid.grid.lat[i] as f64, grid.grid.lon[i] as f64);
                w = w.min(lo);
                e = e.max(lo);
                s = s.min(la);
                n = n.max(la);
            }
            let cols: Vec<Result<RawColumn, String>> = paths
                .par_iter()
                .map(|(_, p)| {
                    let r = st.store.reader(p)?;
                    Store::box_column(&r, &cells)
                })
                .collect();
            Ok(Columns {
                members: paths.iter().map(|x| x.0.clone()).collect(),
                cols: cols.into_iter().collect::<Result<Vec<_>, _>>()?,
                lat: (s + n) / 2.0,
                lon: (w + e) / 2.0,
                fx: None,
                fy: None,
                footprint: Some((s, w, n, e)),
                cells: cells.len(),
                label: format!("{label} ({} pts)", cells.len()),
                read_ms: t.elapsed().as_secs_f64() * 1e3,
            })
        }
    }
}

fn meta_for(cycle: &str, lead: u32, c: &Columns, sample: &str) -> SoundingMetadata {
    let v = valid_iso(cycle, lead);
    SoundingMetadata {
        station_id: c.label.clone(),
        valid_time: format!("{} {}Z", &v[..10], &v[11..16]),
        latitude_deg: Some(c.lat),
        longitude_deg: Some(c.lon),
        elevation_m: None,
        sample_method: Some(sample.to_string()),
        box_radius_lat_deg: c.footprint.map(|f| (f.2 - f.0) / 2.0),
        box_radius_lon_deg: c.footprint.map(|f| (f.3 - f.1) / 2.0),
    }
}

fn build_analysis(
    raw: &RawColumn,
    meta: SoundingMetadata,
    ao: &analysis::AnalysisOpts,
    fp: Option<(f64, f64, f64, f64)>,
) -> Result<Analysis, String> {
    let col = analysis::to_column(raw, meta)?;
    let mut a = analysis::analyze(col, ao)?;
    if let Some((s, w, n, e)) = fp {
        a.prof
            .set_location_footprint(sharppyrs::LocationFootprint::new(s, w, n, e));
    }
    Ok(a)
}

pub(crate) fn patches(d: &DisplayOpts, a: &Analysis) -> sharppyrs::NativeDiagnosticPatchBoard {
    let mut p = render::base_patches();
    for (k, id) in &d.blank {
        p.push(sharppyrs::NativeDiagnosticPatch::blank(*k, id.clone()));
    }
    if let Some(label) = &a.sm_label {
        let (u, v) = (a.prof.srwind.0, a.prof.srwind.1);
        let dir = (-u).atan2(-v).to_degrees().rem_euclid(360.0);
        p.push(sharppyrs::NativeDiagnosticPatch::replace(
            sharppyrs::DiagnosticTablePanelKind::Kinematics,
            "kin.bunkers_right",
            sharppyrs::DiagnosticTableRow::new(
                label.clone(),
                format!("{:.0}/{:.0} kt", dir, u.hypot(v)),
            ),
        ));
    }
    sharppyrs::NativeDiagnosticPatchBoard { patches: p }
}

/// Resolved cycle + lead + member selection.
struct Sel {
    cycle: String,
    lead: u32,
    /// "memNNN" | "mean" | "all"
    member: String,
    /// (member label, file) to read
    paths: Vec<(String, PathBuf)>,
}

fn norm_member(m: &str) -> String {
    if let Ok(n) = m.parse::<u32>() {
        format!("mem{n:03}")
    } else {
        m.to_string()
    }
}

fn resolve_cycle(st: &AppState, q: &Q) -> R<(String, Arc<store::Catalog>)> {
    let cat = st.store.catalog();
    let cycle = match qs(q, "cycle") {
        None | Some("latest") => cat
            .keys()
            .next_back()
            .cloned()
            .ok_or_else(|| ApiErr::NotFound("store is empty".into(), json!([])))?,
        Some(c) => c.to_string(),
    };
    if !cat.contains_key(&cycle) {
        return Err(ApiErr::NotFound(
            format!("cycle {cycle} not in store"),
            json!(cat.keys().rev().collect::<Vec<_>>()),
        ));
    }
    Ok((cycle, cat))
}

/// Contract 404s for a missing cycle / lead / member. `mean` uses the
/// store's own mean member when present, else the per-level mean of members.
fn select(st: &AppState, q: &Q, default_member: &str, lead_opt: Option<u32>) -> R<Sel> {
    let (cycle, cat) = resolve_cycle(st, q)?;
    let mm = &cat[&cycle];
    let mut leads: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for (m, ls) in mm {
        for l in ls.keys() {
            leads.entry(*l).or_default().push(m.clone());
        }
    }
    let lead = match lead_opt.or(qf(q, "lead")?.map(|l| l as u32)) {
        Some(l) => l,
        None => *leads
            .keys()
            .next()
            .ok_or_else(|| ApiErr::NotFound("cycle has no leads".into(), json!([])))?,
    };
    let have = leads.get(&lead).ok_or_else(|| {
        ApiErr::NotFound(
            format!("lead {lead} not in cycle {cycle}"),
            json!(leads.keys().collect::<Vec<_>>()),
        )
    })?;
    let real: Vec<String> = have
        .iter()
        .filter(|m| m.as_str() != "mean")
        .cloned()
        .collect();
    let subset: Vec<String> = match qs(q, "members") {
        Some(s) => s.split(',').map(|m| norm_member(m.trim())).collect(),
        None => real.clone(),
    };
    for m in &subset {
        if !real.contains(m) {
            return Err(ApiErr::NotFound(
                format!("member {m} not in {cycle} +{lead}"),
                json!(have),
            ));
        }
    }
    let member = norm_member(qs(q, "member").unwrap_or(default_member));
    let path = |m: &str| (m.to_string(), mm[m][&lead].clone());
    let paths = match member.as_str() {
        "mean" => {
            if have.iter().any(|m| m == "mean") && qs(q, "members").is_none() {
                vec![path("mean")]
            } else if !subset.is_empty() {
                subset.iter().map(|m| path(m)).collect()
            } else {
                return Err(ApiErr::NotFound(
                    format!("no members or mean in {cycle} +{lead}"),
                    json!(have),
                ));
            }
        }
        "all" => {
            if subset.is_empty() {
                return Err(ApiErr::NotFound(
                    format!("no members landed for {cycle} +{lead} (only the mean)"),
                    json!(have),
                ));
            }
            subset.iter().map(|m| path(m)).collect()
        }
        m => {
            if !have.iter().any(|h| h == m) {
                return Err(ApiErr::NotFound(
                    format!("member {m} not in {cycle} +{lead}"),
                    json!(have),
                ));
            }
            vec![path(m)]
        }
    };
    Ok(Sel {
        cycle,
        lead,
        member,
        paths,
    })
}

fn where_json(c: &Columns, wh: &Where, method: &str) -> Value {
    let elev = c
        .cols
        .first()
        .map(|x| x.sfc.orog_m)
        .filter(|x| x.is_finite());
    json!({
        "kind": if matches!(wh, Where::Point { .. }) { "point" } else { "box" },
        "lat": (c.lat * 1e4).round() / 1e4, "lon": (c.lon * 1e4).round() / 1e4,
        "i": c.fx.map(|x| (x * 100.0).round() / 100.0), "j": c.fy.map(|x| (x * 100.0).round() / 100.0),
        "elev_m": elev.map(|x| x.round()),
        "box": match wh { Where::Area { bbox: Some(b), .. } => json!(b), Where::Area { poly, .. } => json!(poly), _ => Value::Null },
        "footprint": c.footprint.map(|(s, w, n, e)| json!([w, s, e, n])),
        "cells": c.cells, "method": method, "label": c.label,
    })
}

fn levels_json(a: &Analysis) -> Value {
    let c = &a.column;
    let r = |x: f64, m: f64| {
        if x.is_finite() {
            json!((x * m).round() / m)
        } else {
            Value::Null
        }
    };
    Value::Array(
        (0..c.len())
            .map(|i| {
                let (u, v) = (c.u_ms[i], c.v_ms[i]);
                let wdir = (-u).atan2(-v).to_degrees().rem_euclid(360.0);
                json!({"p": r(c.pressure_hpa[i], 100.0), "z": r(c.height_m_msl[i], 10.0), "t": r(c.temperature_c[i], 100.0),
                       "td": r(c.dewpoint_c[i], 100.0), "wdir": r(wdir, 10.0), "wspd": r(u.hypot(v) * 1.943_844_5, 10.0),
                       "omega": c.omega_pa_s.get(i).map(|x| r(*x, 10000.0)).unwrap_or(Value::Null)})
            })
            .collect(),
    )
}

fn parcels_json(a: &Analysis) -> Value {
    let p = &a.prof;
    let r = |x: f64| {
        if x.is_finite() {
            json!(x.round())
        } else {
            Value::Null
        }
    };
    let agl = |h: f64| h; // sharprs parcel heights are already AGL
    let one = |pc: &sharppyrs::Parcel| {
        json!({"cape": r(pc.bplus), "cin": r(pc.bminus), "lcl_m": r(agl(pc.lclhght)), "lfc_m": r(agl(pc.lfchght)),
               "el_m": r(agl(pc.elhght)), "li": if pc.li5.is_finite() { json!((pc.li5 * 10.0).round() / 10.0) } else { Value::Null },
               "lpl_hpa": r(pc.pres)})
    };
    let mut v = json!({"sb": one(&p.sfcpcl), "ml": one(&p.mlpcl), "mu": one(&p.mupcl), "fcst": one(&p.fcstpcl)});
    if a.user_parcel {
        v["user"] = v["fcst"].clone();
    }
    v
}

fn sm_json(a: &Analysis, q: &Q) -> Value {
    let (u, v) = (a.prof.srwind.0, a.prof.srwind.1);
    json!({"method": qs(q, "sm").unwrap_or("bunkers_rm"), "label": a.sm_label.clone().unwrap_or_else(|| "Bunkers right".into()),
           "dir": ((-u).atan2(-v).to_degrees().rem_euclid(360.0)).round(), "spd_kt": u.hypot(v).round()})
}

fn key_json(a: &Analysis) -> Value {
    let d = &a.derived;
    let r = |x: f64, m: f64| {
        if x.is_finite() {
            json!((x * m).round() / m)
        } else {
            Value::Null
        }
    };
    json!({"mlcape": r(a.prof.mlpcl.bplus, 1.0), "mucape": r(a.prof.mupcl.bplus, 1.0), "sbcape": r(a.prof.sfcpcl.bplus, 1.0),
           "mlcin": r(a.prof.mlpcl.bminus, 1.0), "srh01": r(d.srh1km, 1.0), "srh03": r(d.srh3km, 1.0), "esrh": r(d.right_esrh, 1.0),
           "shr06": r(d.sfc_6km_shear.0.hypot(d.sfc_6km_shear.1), 1.0), "ebwd": r(d.ebwd.0.hypot(d.ebwd.1), 1.0),
           "stp": r(d.stp_cin, 100.0), "stp_fixed": r(d.stp_fixed, 100.0), "scp": r(d.right_scp, 100.0), "ship": r(d.ship, 100.0),
           "pwat": r(d.pwat, 100.0), "dcape": r(d.dcape, 1.0)})
}

fn spread_json(rows: &[&Analysis]) -> Value {
    let keys = [
        "mlcape", "mucape", "sbcape", "srh01", "srh03", "esrh", "shr06", "stp", "scp",
    ];
    let ks: Vec<Value> = rows.iter().map(|a| key_json(a)).collect();
    let mut m = serde_json::Map::new();
    let r = |x: f64| {
        if x.is_finite() {
            json!((x * 100.0).round() / 100.0)
        } else {
            Value::Null
        }
    };
    for k in keys {
        let v: Vec<f64> = ks.iter().filter_map(|x| x[k].as_f64()).collect();
        let pq = |q: f64| r(store::percentile(&mut v.clone(), q));
        m.insert(k.to_string(), json!({"min": pq(0.0), "p10": pq(10.0), "p50": pq(50.0), "p90": pq(90.0), "max": pq(100.0), "n": v.len()}));
    }
    Value::Object(m)
}

pub(crate) fn bands_from(cols: &[RawColumn], members: &[&Analysis]) -> Bands {
    let nl = cols.iter().map(|c| c.levels()).min().unwrap_or(0);
    let qs = [10.0, 25.0, 50.0, 75.0, 90.0];
    let mut b = Bands::default();
    for k in 0..nl {
        let pv: Vec<f64> = cols
            .iter()
            .map(|c| c.p[k])
            .filter(|x| x.is_finite())
            .collect();
        b.p.push(if pv.is_empty() {
            f64::NAN
        } else {
            pv.iter().sum::<f64>() / pv.len() as f64
        });
        for (qi, q) in qs.iter().enumerate() {
            b.t[qi].push(store::percentile(
                &mut cols.iter().map(|c| c.t[k]).collect(),
                *q,
            ));
            b.td[qi].push(store::percentile(
                &mut cols.iter().map(|c| c.td[k]).collect(),
                *q,
            ));
        }
    }
    // median hodograph on a common 0-10 km AGL ladder
    for i in 0..=40 {
        let z = i as f64 * 250.0;
        let mut us = vec![];
        let mut vs = vec![];
        for a in members {
            let inner = &a.prof.inner;
            let p = inner.pres_at_height(inner.to_msl(z));
            if p.is_finite() {
                let (u, v) = inner.interp_wind(p);
                us.push(u);
                vs.push(v);
            }
        }
        b.hodo_med.push((
            store::percentile(&mut us, 50.0),
            store::percentile(&mut vs, 50.0),
        ));
    }
    b
}

pub(crate) fn hodo_trace(a: &Analysis) -> Vec<(f64, f64)> {
    let inner = &a.prof.inner;
    (0..inner.pres.len())
        .filter(|&i| inner.to_agl(inner.hght[i]) <= 10_000.0)
        .map(|i| (inner.u[i], inner.v[i]))
        .collect()
}

fn prepare(st: &AppState, kind: Kind, fmt: &str, q: &Q) -> R<Prepared> {
    let t0 = Instant::now();
    let ao = options::parse_analysis(q)?;
    let dopt = options::parse_display(q)?;
    let wh = parse_where(q)?;
    if kind == Kind::TimeHeight {
        return prepare_timeheight(st, fmt, q, &ao, &dopt, &wh);
    }
    let default_member = if kind == Kind::Ensemble {
        "all"
    } else {
        "mean"
    };
    let mut sel = select(st, q, default_member, None)?;
    if kind == Kind::Ensemble && sel.member != "all" {
        // ensemble views always use every member (members= narrows)
        let mut q2 = q.clone();
        q2.insert("member".into(), "all".into());
        sel = select(st, &q2, "all", Some(sel.lead))?;
    }
    if kind == Kind::Sounding && fmt == "png" && sel.member == "all" {
        return Err("member=all is drawn by /api/ensemble.png".into());
    }
    let legacy = q.contains_key("__legacy");
    let kname = match kind {
        Kind::Sounding => format!("sounding.{fmt}"),
        Kind::Ensemble => format!("ensemble.{fmt}"),
        Kind::TimeHeight => unreachable!(),
    };
    let key = cache_key(&kname, &sel.cycle, &sel.lead.to_string(), q);
    if let Some(c) = st.cache.lock().unwrap().get(&key) {
        return Ok(Prepared::Cached(key, c));
    }
    let cols = read_columns(st, &sel.paths, &wh)?;
    let t_an = Instant::now();
    let multi = cols.cols.len() > 1;
    let stored_mean = sel.member == "mean"
        && !multi
        && cols.members.first().map(|m| m == "mean").unwrap_or(false);
    let method = match (&wh, multi) {
        (Where::Point { .. }, false) => "bilinear".to_string(),
        (Where::Point { .. }, true) => format!("bilinear, mean of {} members", cols.cols.len()),
        (Where::Area { .. }, false) => format!("area mean of {} cells", cols.cells),
        (Where::Area { .. }, true) => format!(
            "area mean of {} cells, mean of {} members",
            cols.cells,
            cols.cols.len()
        ),
    };
    let method = if stored_mean {
        format!("{method}, stored ensemble-mean fields")
    } else {
        method
    };
    let display_raw = if multi {
        store::mean_columns(&cols.cols)?
    } else {
        cols.cols[0].clone()
    };
    let meta = meta_for(&sel.cycle, sel.lead, &cols, &method);
    let main = build_analysis(&display_raw, meta.clone(), &ao, cols.footprint)?;
    let member_an: Vec<(String, Result<Analysis, String>)> = if multi && sel.member == "all" {
        cols.members
            .par_iter()
            .zip(cols.cols.par_iter())
            .map(|(m, c)| (m.clone(), build_analysis(c, meta.clone(), &ao, None)))
            .collect()
    } else {
        vec![]
    };
    let an_ms = t_an.elapsed().as_secs_f64() * 1e3;
    let mtag = match sel.member.as_str() {
        "all" => format!("mean + {} members", cols.members.len()),
        "mean" if multi => format!("mean of {}", cols.members.len()),
        "mean" => "ensemble mean".into(),
        m => m.to_string(),
    };
    let title = dopt.title.clone().unwrap_or_else(|| {
        format!(
            "{} {}Z +{}m  valid {}  {}  [{}]",
            st.store.label(&sel.cycle).to_uppercase(),
            &sel.cycle[8..12],
            sel.lead,
            &meta.valid_time[11..],
            cols.label,
            mtag
        )
    });
    let wj = where_json(&cols, &wh, &method);
    let mut hdrs = vec![
        ("x-sounding-where".to_string(), wj.to_string()),
        (
            "x-sounding-timing".to_string(),
            format!(
                "read={:.2}ms; analysis={an_ms:.2}ms; members={}; cells={}",
                cols.read_ms,
                cols.members.len(),
                cols.cells
            ),
        ),
    ];
    let blank_ids: Vec<String> = dopt.blank.iter().map(|(_, id)| id.clone()).collect();

    if fmt == "txt" {
        let (body, ext, ctype) = match qs(q, "fmt").unwrap_or("sharppy") {
            "sharppy" | "raw" => (
                sharppy_raw(&main.column, &title),
                "txt",
                "text/plain; charset=utf-8",
            ),
            "csv" => (
                csv(&main.column, &wj, &meta),
                "csv",
                "text/csv; charset=utf-8",
            ),
            f => return Err(format!("fmt must be sharppy|csv, got '{f}'").into()),
        };
        let place: String = cols
            .label
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let fname = format!(
            "{}_{}_f{:03}_{}_{}.{ext}",
            st.store.label(&sel.cycle),
            sel.cycle,
            sel.lead,
            sel.member,
            place
        );
        hdrs.push((
            "content-disposition".into(),
            format!("attachment; filename=\"{fname}\""),
        ));
        hdrs.push((
            "x-sounding-ms".into(),
            format!("{:.1}", t0.elapsed().as_secs_f64() * 1e3),
        ));
        return Ok(Prepared::Bytes(key, body.into_bytes(), ctype, hdrs));
    }
    if fmt == "json" && legacy {
        let mut v = analysis_json(&main);
        v["request"] = json!({
            "endpoint": if matches!(wh, Where::Point { .. }) { "point" } else { "box" },
            "model": st.store.label(&sel.cycle), "cycle": sel.cycle, "lead_min": sel.lead, "valid": meta.valid_time,
            "member_mode": match sel.member.as_str() { "all" => "all", "mean" => "mean", _ => "one" },
            "members": cols.members, "sample_method": method, "cells": cols.cells,
            "location": {"lat": cols.lat, "lon": cols.lon, "label": cols.label},
            "footprint": cols.footprint.map(|(s, w, n, e)| json!({"south": s, "west": w, "north": n, "east": e})),
            "title": title, "where": wj,
        });
        v["tables"] = tables::tables_json(&main, &blank_ids);
        if multi {
            v["members"] = Value::Array(
                member_an
                    .iter()
                    .map(|(m, a)| match a {
                        Ok(a) => json!({"member": m, "indices": analysis::summary_json(a),
                            "t_c": a.column.temperature_c, "td_c": a.column.dewpoint_c, "p_hpa": a.column.pressure_hpa,
                            "hodo": a.prof.inner.u.iter().zip(&a.prof.inner.v).zip(&a.prof.inner.hght)
                                .filter(|((u, v), _)| u.is_finite() && v.is_finite())
                                .map(|((u, v), z)| json!([(a.prof.inner.to_agl(*z)).round(), (u * 10.0).round() / 10.0, (v * 10.0).round() / 10.0]))
                                .collect::<Vec<_>>()}),
                        Err(e) => json!({"member": m, "error": e}),
                    })
                    .collect(),
            );
            let (lo, hi) = qs(q, "pct")
                .and_then(|s| {
                    let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                    (v.len() == 2 && v[0] < v[1]).then(|| (v[0], v[1]))
                })
                .unwrap_or((10.0, 90.0));
            v["envelope"] = json!(store::envelope(&cols.cols, lo, hi));
            v["percentiles"] = store::percentiles5(&cols.cols);
        }
        v["timing_ms"] = json!({"read": cols.read_ms, "analysis": an_ms, "total_server": t0.elapsed().as_secs_f64() * 1e3});
        hdrs.push((
            "x-sounding-ms".into(),
            format!("{:.1}", t0.elapsed().as_secs_f64() * 1e3),
        ));
        return Ok(Prepared::Bytes(
            format!("legacy|{key}"),
            v.to_string().into_bytes(),
            "application/json",
            hdrs,
        ));
    }
    if fmt == "json" {
        let mut v = json!({
            "cycle": sel.cycle, "lead": sel.lead, "valid": valid_iso(&sel.cycle, sel.lead), "member": sel.member,
            "members_used": cols.members, "title": title,
            "where": wj,
            "levels": levels_json(&main),
            "tables": tables::tables_json(&main, &blank_ids),
            "parcels": parcels_json(&main),
            "storm_motion": sm_json(&main, q),
            "notes": main.notes,
            "analysis": analysis_json(&main),
        });
        if sel.member == "all" {
            let ok: Vec<&Analysis> = member_an
                .iter()
                .filter_map(|(_, a)| a.as_ref().ok())
                .collect();
            v["members"] = Value::Array(
                member_an
                    .iter()
                    .map(|(m, a)| match a {
                        Ok(a) => json!({"member": m, "levels": levels_json(a), "key": key_json(a),
                                        "hodo": hodo_trace(a).iter().map(|(u, v)| json!([(u * 10.0).round() / 10.0, (v * 10.0).round() / 10.0])).collect::<Vec<_>>()}),
                        Err(e) => json!({"member": m, "error": e}),
                    })
                    .collect(),
            );
            v["percentiles"] = store::percentiles5(&cols.cols);
            v["spread"] = spread_json(&ok);
        }
        v["ms"] = json!({"read": cols.read_ms, "analysis": an_ms, "total": t0.elapsed().as_secs_f64() * 1e3});
        hdrs.push((
            "x-sounding-ms".into(),
            format!("{:.1}", t0.elapsed().as_secs_f64() * 1e3),
        ));
        return Ok(Prepared::Bytes(
            key,
            v.to_string().into_bytes(),
            "application/json",
            hdrs,
        ));
    }

    let overlay = if kind == Kind::Ensemble {
        let ok: Vec<&Analysis> = member_an
            .iter()
            .filter_map(|(_, a)| a.as_ref().ok())
            .collect();
        let envelope = qs(q, "mode") == Some("envelope");
        let members: Vec<MemberTrace> = if envelope {
            vec![]
        } else {
            member_an
                .iter()
                .filter_map(|(m, a)| {
                    a.as_ref().ok().map(|a| MemberTrace {
                        label: m.clone(),
                        p: a.column.pressure_hpa.clone(),
                        t: a.column.temperature_c.clone(),
                        td: a.column.dewpoint_c.clone(),
                        hodo: hodo_trace(a),
                    })
                })
                .collect()
        };
        let legend = if envelope {
            format!(
                "{} members · p10-p90 and p25-p75 bands, median · thick = ensemble mean",
                cols.members.len()
            )
        } else {
            format!(
                "{} members (thin) · p10-p90 band · thick = ensemble mean",
                cols.members.len()
            )
        };
        Some(Overlay {
            members,
            bands: if qs(q, "envelope") == Some("off") {
                None
            } else {
                Some(bands_from(&cols.cols, &ok))
            },
            inner_band: envelope,
            show_t: dopt.spag_t,
            show_td: dopt.spag_td,
            alpha: dopt.spag_alpha,
            legend,
        })
    } else {
        None
    };
    let scene = SoundingScene {
        patches: patches(&dopt, &main),
        analysis: Arc::new(main),
        view: dopt.view.clone(),
        layout: dopt.layout.clone(),
        corner: dopt.corner,
        parcel: dopt.parcel,
        title,
        brand: dopt.brand.clone(),
        style: dopt.style.clone(),
        overlay,
    };
    Ok(Prepared::Render(
        key,
        Arc::new(Job::Sounding(scene)),
        Vec2::new(dopt.width, dopt.height),
        dopt.ppp,
        hdrs,
        an_ms,
    ))
}

/// SPC/SHARPpy %RAW% text, the bowecho `sharppy_raw_text` format.
fn sharppy_raw(c: &rustwx_sounding::SoundingColumn, title: &str) -> String {
    let mut out = format!(
        "%TITLE%\n{}\n\n   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD\n-------------------------------------------------------------------\n%RAW%\n",
        title.replace(['\n', '\r'], " ")
    );
    for i in 0..c.len() {
        let (u, v) = (c.u_ms[i], c.v_ms[i]);
        let spd = u.hypot(v);
        let dir = if spd <= 1e-12 {
            0.0
        } else {
            (-u).atan2(-v).to_degrees().rem_euclid(360.0)
        };
        out.push_str(&format!(
            "{:10.3}, {:10.3}, {:10.3}, {:10.3}, {:10.3}, {:10.3}\n",
            c.pressure_hpa[i],
            c.height_m_msl[i],
            c.temperature_c[i],
            c.dewpoint_c[i],
            dir,
            spd * 1.943_844_492_440_604_7
        ));
    }
    out.push_str("%END%\n");
    out
}

/// Levels CSV (bowecho `corrected_profile_csv` columns) after a header comment.
fn csv(c: &rustwx_sounding::SoundingColumn, wj: &Value, meta: &SoundingMetadata) -> String {
    let mut out = format!(
        "# {} valid {} where {}\n",
        meta.station_id, meta.valid_time, wj
    );
    out.push_str("pressure_hpa,height_m_msl,height_m_agl,temperature_c,dewpoint_c,u_ms,v_ms,wind_direction_deg,wind_speed_kt,omega_pa_s\n");
    let z0 = c.height_m_msl[0];
    for i in 0..c.len() {
        let (u, v) = (c.u_ms[i], c.v_ms[i]);
        let spd = u.hypot(v);
        let dir = if spd <= 1e-12 {
            0.0
        } else {
            (-u).atan2(-v).to_degrees().rem_euclid(360.0)
        };
        let om = c
            .omega_pa_s
            .get(i)
            .map(|x| format!("{x:.6}"))
            .unwrap_or_default();
        out.push_str(&format!(
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            c.pressure_hpa[i],
            c.height_m_msl[i],
            c.height_m_msl[i] - z0,
            c.temperature_c[i],
            c.dewpoint_c[i],
            u,
            v,
            dir,
            spd * 1.943_844_492_440_604_7,
            om
        ));
    }
    out
}

fn prepare_timeheight(
    st: &AppState,
    fmt: &str,
    q: &Q,
    ao: &analysis::AnalysisOpts,
    dopt: &DisplayOpts,
    wh: &Where,
) -> R<Prepared> {
    let (cycle, cat) = resolve_cycle(st, q)?;
    let mm = &cat[&cycle];
    let mut all: Vec<u32> = mm.values().flat_map(|ls| ls.keys().copied()).collect();
    all.sort();
    all.dedup();
    let mut leads = all;
    if let Some(s) = qs(q, "leads") {
        let want: Vec<u32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        leads.retain(|l| want.contains(l));
    }
    let key = cache_key(
        &format!("timeheight.{fmt}"),
        &cycle,
        &format!("{leads:?}"),
        q,
    );
    if let Some(c) = st.cache.lock().unwrap().get(&key) {
        return Ok(Prepared::Cached(key, c));
    }
    let field = qs(q, "field").unwrap_or("rh").to_string();
    let (ramp, flabel) = ramp_for(&field)?;
    let ztop = qf(q, "ztop")?.unwrap_or(10.0).clamp(1.0, 16.0);
    let barbs_on = options::qb(q, "barbs", true)?;
    let mut per_lead: Vec<(u32, Analysis, Columns)> = vec![];
    let mut read_ms = 0.0;
    let mut an_ms = 0.0;
    let mut skipped = vec![];
    for l in &leads {
        let mut q2 = q.clone();
        q2.insert("lead".into(), l.to_string());
        q2.insert("cycle".into(), cycle.clone());
        let Ok(sel) = select(st, &q2, "mean", Some(*l)) else {
            skipped.push(*l);
            continue;
        };
        if sel.member == "all" {
            return Err("timeheight takes member=memNNN or mean".into());
        }
        let cols = read_columns(st, &sel.paths, wh)?;
        read_ms += cols.read_ms;
        let ta = Instant::now();
        let raw = if cols.cols.len() > 1 {
            store::mean_columns(&cols.cols)?
        } else {
            cols.cols[0].clone()
        };
        let meta = meta_for(&cycle, *l, &cols, "time-height");
        let a = build_analysis(&raw, meta, ao, cols.footprint)?;
        an_ms += ta.elapsed().as_secs_f64() * 1e3;
        per_lead.push((*l, a, cols));
    }
    if per_lead.is_empty() {
        return Err(ApiErr::NotFound(
            format!("member not available at any lead of {cycle}"),
            json!(leads),
        ));
    }
    let leads: Vec<u32> = per_lead.iter().map(|x| x.0).collect();
    let label = per_lead[0].2.label.clone();
    let member = norm_member(qs(q, "member").unwrap_or("mean"));
    let dz = 0.1;
    let nz = (ztop / dz).round() as usize + 1;
    let z_km: Vec<f64> = (0..nz).map(|i| i as f64 * dz).collect();
    let sample = |a: &Analysis, zk: f64| -> f64 {
        let inner = &a.prof.inner;
        if field == "srh" {
            if zk <= 0.0 {
                return 0.0;
            }
            return sharppyrs::extras::helicity(
                inner,
                0.0,
                zk * 1000.0,
                a.prof.srwind.0,
                a.prof.srwind.1,
            )
            .0;
        }
        let target = inner.to_msl(zk * 1000.0);
        let h = &inner.hght;
        let val = |k: usize| -> f64 {
            match field.as_str() {
                "rh" => inner.relh.get(k).copied().unwrap_or(f64::NAN),
                "thetae" => inner.thetae.get(k).copied().unwrap_or(f64::NAN),
                "t" => inner.tmpc[k],
                "td" => inner.dwpc[k],
                "tdd" => inner.tmpc[k] - inner.dwpc[k],
                "wspd" => inner.wspd[k],
                "omega" => inner.omeg.get(k).copied().unwrap_or(f64::NAN) * 10.0,
                "tw" => inner.wetbulb.get(k).copied().unwrap_or(f64::NAN),
                _ => f64::NAN,
            }
        };
        match h.iter().position(|x| *x >= target) {
            Some(0) => val(0),
            Some(k) => {
                let w = (target - h[k - 1]) / (h[k] - h[k - 1]);
                val(k - 1) + (val(k) - val(k - 1)) * w
            }
            None => f64::NAN,
        }
    };
    let field_vals: Vec<Vec<f64>> = per_lead
        .iter()
        .map(|(_, a, _)| z_km.iter().map(|z| sample(a, *z)).collect())
        .collect();
    let mut hdrs = vec![(
        "x-sounding-timing".to_string(),
        format!(
            "read={read_ms:.2}ms; analysis={an_ms:.2}ms; leads={}",
            leads.len()
        ),
    )];
    let legacy = q.contains_key("__legacy");
    if fmt == "json" && legacy {
        let frames: Vec<String> = leads
            .iter()
            .map(|l| {
                let mut items: Vec<(String, String)> = q
                    .iter()
                    .filter(|(k, _)| {
                        !matches!(
                            k.as_str(),
                            "lead" | "leads" | "field" | "ztop" | "barbs" | "cycle" | "__legacy"
                        )
                    })
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                items.sort();
                items.push(("cycle".into(), cycle.clone()));
                items.push(("lead".into(), l.to_string()));
                let ep = if matches!(wh, Where::Area { .. }) {
                    "box"
                } else {
                    "point"
                };
                format!(
                    "/api/sounding/{ep}.png?{}",
                    items
                        .iter()
                        .map(|(k, v)| format!("{k}={}", urlenc(v)))
                        .collect::<Vec<_>>()
                        .join("&")
                )
            })
            .collect();
        let v = json!({
            "request": {"endpoint": "series", "cycle": cycle, "member_mode": member, "location": label, "field": field, "ztop_km": ztop},
            "leads_min": leads, "skipped_leads": skipped, "z_km": z_km,
            "field": field_vals,
            "per_lead": per_lead.iter().map(|(l, a, _)| json!({"lead_min": l, "valid": a.column.metadata.valid_time, "indices": analysis::summary_json(a), "analysis": analysis_json(a)})).collect::<Vec<_>>(),
            "frames": frames,
            "timing_ms": {"read": read_ms, "analysis": an_ms},
        });
        return Ok(Prepared::Bytes(
            format!("legacy|{key}"),
            v.to_string().into_bytes(),
            "application/json",
            hdrs,
        ));
    }
    if fmt == "json" {
        let r = |x: f64| {
            if x.is_finite() {
                json!((x * 100.0).round() / 100.0)
            } else {
                Value::Null
            }
        };
        let frames: Vec<String> = leads
            .iter()
            .map(|l| {
                let mut items: Vec<(String, String)> = q
                    .iter()
                    .filter(|(k, _)| {
                        !matches!(
                            k.as_str(),
                            "lead" | "leads" | "field" | "ztop" | "barbs" | "cycle" | "__legacy"
                        )
                    })
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                items.sort();
                items.push(("cycle".into(), cycle.clone()));
                items.push(("lead".into(), l.to_string()));
                format!(
                    "/api/sounding.png?{}",
                    items
                        .iter()
                        .map(|(k, v)| format!("{k}={}", urlenc(v)))
                        .collect::<Vec<_>>()
                        .join("&")
                )
            })
            .collect();
        let v = json!({
            "cycle": cycle, "member": member, "field": field, "units": flabel, "ztop_km": ztop, "where": label,
            "leads": leads, "skipped_leads": skipped, "z_km": z_km,
            "values": field_vals.iter().map(|c| c.iter().map(|x| r(*x)).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "key": per_lead.iter().map(|(l, a, _)| json!({"lead": l, "valid": valid_iso(&cycle, *l), "key": key_json(a)})).collect::<Vec<_>>(),
            "frames": frames,
        });
        hdrs.push(("x-sounding-ms".into(), format!("{:.1}", read_ms + an_ms)));
        return Ok(Prepared::Bytes(
            key,
            v.to_string().into_bytes(),
            "application/json",
            hdrs,
        ));
    }
    let mut barbs = vec![];
    if barbs_on {
        let step = if ztop > 6.0 { 1.0 } else { 0.5 };
        let bz: Vec<f64> = (0..)
            .map(|i| 0.25 + i as f64 * step)
            .take_while(|z| *z <= ztop)
            .collect();
        for (i, (_, a, _)) in per_lead.iter().enumerate() {
            let inner = &a.prof.inner;
            for z in &bz {
                let p = inner.pres_at_height(inner.to_msl(z * 1000.0));
                if p.is_finite() {
                    let (d, s) = inner.interp_vec(p);
                    barbs.push((i, *z, d, s));
                }
            }
        }
    }
    let agl = |_a: &Analysis, h: f64| h / 1000.0; // parcel heights are already AGL (m)
    let lines = vec![
        (
            "ML LCL".to_string(),
            Color32::from_rgb(80, 220, 80),
            per_lead
                .iter()
                .map(|(_, a, _)| agl(a, a.prof.mlpcl.lclhght))
                .collect(),
        ),
        (
            "MU LFC".to_string(),
            Color32::from_rgb(255, 210, 60),
            per_lead
                .iter()
                .map(|(_, a, _)| agl(a, a.prof.mupcl.lfchght))
                .collect(),
        ),
        (
            "MU EL".to_string(),
            Color32::from_rgb(200, 120, 255),
            per_lead
                .iter()
                .map(|(_, a, _)| agl(a, a.prof.mupcl.elhght))
                .collect(),
        ),
        (
            "Eff top".to_string(),
            Color32::from_rgb(90, 200, 255),
            per_lead
                .iter()
                .map(|(_, a, _)| a.prof.etopm / 1000.0)
                .collect(),
        ),
    ];
    let strip = vec![
        (
            "MLCAPE".to_string(),
            per_lead
                .iter()
                .map(|(_, a, _)| a.prof.mlpcl.bplus)
                .collect(),
        ),
        (
            "MLCIN".to_string(),
            per_lead
                .iter()
                .map(|(_, a, _)| a.prof.mlpcl.bminus)
                .collect(),
        ),
        (
            "SRH 0-1".to_string(),
            per_lead.iter().map(|(_, a, _)| a.derived.srh1km).collect(),
        ),
        (
            "ESRH".to_string(),
            per_lead
                .iter()
                .map(|(_, a, _)| a.derived.right_esrh)
                .collect(),
        ),
        (
            "STP".to_string(),
            per_lead.iter().map(|(_, a, _)| a.derived.stp_cin).collect(),
        ),
    ];
    let mtag = if member == "mean" {
        "ensemble mean".to_string()
    } else {
        member.clone()
    };
    let scene = TimeHeightScene {
        title: dopt.title.clone().unwrap_or_else(|| {
            format!(
                "{} {}Z cycle  time-height  {}  [{}]",
                st.store.label(&cycle).to_uppercase(),
                &cycle[8..12],
                label,
                mtag
            )
        }),
        subtitle: format!(
            "{flabel} shaded · wind barbs (kt) · lines: ML LCL, MU LFC, MU EL, effective inflow top"
        ),
        leads_min: leads,
        z_km,
        field: field_vals,
        field_label: flabel,
        ramp,
        barbs,
        lines,
        strip,
        style: dopt.style.clone(),
    };
    let w = qf(q, "w")?.unwrap_or(1100.0).clamp(400.0, 4000.0) as f32;
    let h = qf(q, "h")?.unwrap_or(800.0).clamp(300.0, 4000.0) as f32;
    Ok(Prepared::Render(
        key,
        Arc::new(Job::TimeHeight(scene)),
        Vec2::new(w, h),
        dopt.ppp,
        hdrs,
        read_ms + an_ms,
    ))
}

fn urlenc(v: &str) -> String {
    v.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b',' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn ramp_for(field: &str) -> R<(Ramp, String)> {
    let c = Color32::from_rgb;
    Ok(match field {
        "rh" => (
            Ramp {
                stops: vec![
                    (0.0, c(120, 70, 20)),
                    (30.0, c(190, 150, 90)),
                    (60.0, c(200, 200, 200)),
                    (80.0, c(90, 190, 110)),
                    (90.0, c(30, 140, 80)),
                    (100.0, c(10, 80, 60)),
                ],
            },
            "RH (%)".into(),
        ),
        "thetae" => (
            Ramp {
                stops: vec![
                    (300.0, c(40, 40, 140)),
                    (320.0, c(60, 120, 220)),
                    (335.0, c(80, 200, 200)),
                    (345.0, c(240, 230, 90)),
                    (355.0, c(240, 140, 50)),
                    (370.0, c(190, 30, 40)),
                ],
            },
            "theta-e (K)".into(),
        ),
        "t" => (
            Ramp {
                stops: vec![
                    (-60.0, c(80, 40, 140)),
                    (-30.0, c(60, 110, 220)),
                    (-10.0, c(120, 200, 240)),
                    (0.0, c(230, 230, 230)),
                    (15.0, c(240, 200, 80)),
                    (35.0, c(200, 40, 40)),
                ],
            },
            "T (C)".into(),
        ),
        "td" => (
            Ramp {
                stops: vec![
                    (-60.0, c(120, 70, 20)),
                    (-30.0, c(190, 150, 90)),
                    (-10.0, c(200, 200, 200)),
                    (5.0, c(120, 200, 130)),
                    (15.0, c(30, 150, 80)),
                    (25.0, c(10, 80, 60)),
                ],
            },
            "Td (C)".into(),
        ),
        "tw" => (
            Ramp {
                stops: vec![
                    (-40.0, c(80, 40, 140)),
                    (-20.0, c(60, 110, 220)),
                    (-5.0, c(120, 200, 240)),
                    (0.0, c(230, 230, 230)),
                    (10.0, c(240, 200, 80)),
                    (25.0, c(200, 40, 40)),
                ],
            },
            "wet-bulb T (C)".into(),
        ),
        "tdd" => (
            Ramp {
                stops: vec![
                    (0.0, c(20, 110, 60)),
                    (3.0, c(90, 190, 110)),
                    (8.0, c(200, 200, 200)),
                    (15.0, c(190, 150, 90)),
                    (30.0, c(120, 70, 20)),
                ],
            },
            "T - Td (C)".into(),
        ),
        "wspd" => (
            Ramp {
                stops: vec![
                    (0.0, c(30, 30, 50)),
                    (20.0, c(60, 100, 200)),
                    (40.0, c(80, 200, 160)),
                    (60.0, c(240, 220, 80)),
                    (80.0, c(240, 120, 40)),
                    (110.0, c(200, 30, 120)),
                ],
            },
            "wind speed (kt)".into(),
        ),
        "omega" => (
            Ramp {
                stops: vec![
                    (-20.0, c(200, 30, 30)),
                    (-5.0, c(240, 160, 140)),
                    (0.0, c(40, 40, 40)),
                    (5.0, c(140, 170, 240)),
                    (20.0, c(30, 60, 200)),
                ],
            },
            "omega (ubar/s)".into(),
        ),
        "srh" => (
            Ramp {
                stops: vec![
                    (-100.0, c(60, 90, 200)),
                    (0.0, c(40, 40, 40)),
                    (100.0, c(90, 170, 90)),
                    (200.0, c(240, 220, 80)),
                    (300.0, c(240, 130, 40)),
                    (500.0, c(200, 30, 120)),
                ],
            },
            "SRH from sfc (m2/s2)".into(),
        ),
        f => {
            return Err(
                format!("field must be t|td|rh|thetae|wspd|omega|srh|tdd|tw, got '{f}'").into(),
            );
        }
    })
}
