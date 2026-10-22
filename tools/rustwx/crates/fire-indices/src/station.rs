//! CWFIS archive filtering and nearest-grid verification. The archive contains
//! indices calculated from station weather, not physical fire observations.
use crate::{PRODUCTS, PendingOutput, Result, hash, publish};
use chrono::{DateTime, NaiveDate};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    archive: PathBuf,
    forecast: Vec<PathBuf>,
    rows: PathBuf,
    receipt: PathBuf,
    start_date: NaiveDate,
    end_date: NaiveDate,
    utc_offset_hours: i32,
    max_distance_km: f64,
    research_station_license: String,
}
#[derive(Default)]
struct Stats {
    n: usize,
    e: f64,
    a: f64,
    q: f64,
    f: f64,
    o: f64,
    ff: f64,
    oo: f64,
    fo: f64,
}
impl Stats {
    fn add(&mut self, f: f64, o: f64) {
        self.n += 1;
        self.e += f - o;
        self.a += (f - o).abs();
        self.q += (f - o).powi(2);
        self.f += f;
        self.o += o;
        self.ff += f * f;
        self.oo += o * o;
        self.fo += f * o;
    }
    fn value(&self) -> serde_json::Value {
        let n = self.n as f64;
        let d = ((n * self.ff - self.f * self.f) * (n * self.oo - self.o * self.o)).sqrt();
        json!({"n":self.n,"bias":self.e/n,"mae":self.a/n,"rmse":(self.q/n).sqrt(),"correlation":if d>0.{Some((n*self.fo-self.f*self.o)/d)}else{None}})
    }
}
pub fn run(path: &Path) -> Result<()> {
    let wall = Instant::now();
    let r: Request = serde_json::from_slice(&fs::read(path)?)?;
    if r.research_station_license != "OGL-Canada-2.0" {
        return Err(
            "station verification requires explicit research_station_license=OGL-Canada-2.0".into(),
        );
    }
    if r.rows.exists() || r.receipt.exists() {
        return Err("preserving existing verification artifacts".into());
    }
    if r.start_date > r.end_date
        || r.forecast.is_empty()
        || !r.max_distance_km.is_finite()
        || r.max_distance_km <= 0.
        || !(-12..=14).contains(&r.utc_offset_hours)
    {
        return Err("invalid verification dates, files, distance or offset".into());
    }
    let mut axes = None;
    let mut days = BTreeMap::new();
    let mut sources = vec![];
    for path in &r.forecast {
        let n = netcrust::open(path)?;
        let lat = n.read_f64("latitude")?;
        let lon = n.read_f64("longitude")?;
        if let Some((y, x)) = &axes {
            if y != &lat || x != &lon {
                return Err("verification forecast grid changed".into());
            }
        } else {
            axes = Some((lat.clone(), lon.clone()));
        }
        let times = n.read_f64("time")?;
        let fields: Vec<_> = PRODUCTS[..6]
            .iter()
            .map(|v| n.read_f64(v))
            .collect::<std::result::Result<_, _>>()?;
        let cells = lat.len() * lon.len();
        if fields.iter().any(|v| v.len() != times.len() * cells) {
            return Err("invalid daily forecast dimensions".into());
        }
        for (i, t) in times.iter().enumerate() {
            let day = (DateTime::from_timestamp(*t as i64, 0).ok_or("bad forecast time")?
                + chrono::Duration::hours(r.utc_offset_hours as i64))
            .date_naive();
            if day >= r.start_date && day <= r.end_date {
                let planes: Vec<Vec<f64>> = fields
                    .iter()
                    .map(|f| f[i * cells..(i + 1) * cells].to_vec())
                    .collect();
                if days.insert(day, planes).is_some() {
                    return Err("duplicate verification forecast day".into());
                }
            }
        }
        sources.push(json!({"path":path,"sha256":hash(path)?}));
    }
    let (lat, lon) = axes.ok_or("no forecast axes")?;
    let decode = Instant::now();
    let mut bytes = Vec::new();
    // Multi-stream bzip2 (the CWFIS archive), capped at 4 GiB of output.
    let mut decoder = std::io::Read::take(
        bzip2::read::MultiBzDecoder::new(fs::File::open(&r.archive)?),
        4 * 1024 * 1024 * 1024u64,
    );
    std::io::Read::read_to_end(&mut decoder, &mut bytes)?;
    let decode_wall = decode.elapsed().as_secs_f64();
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(bytes.as_slice());
    let header = reader.headers()?.clone();
    let col = |name: &str| {
        header
            .iter()
            .position(|s| s == name)
            .ok_or_else(|| format!("station archive missing {name}"))
    };
    let date = col("rep_date")?;
    let name = col("name")?;
    let id = col("aes")?;
    let y = col("lat")?;
    let x = col("lon")?;
    let status = col("calcstatus")?;
    let obs: Vec<_> = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]
        .into_iter()
        .map(col)
        .collect::<std::result::Result<_, _>>()?;
    if let Some(p) = r.rows.parent() {
        fs::create_dir_all(p)?;
    }
    let pending = r.rows.with_file_name(format!(
        "{}.partial",
        r.rows
            .file_name()
            .ok_or("rows require a filename")?
            .to_string_lossy()
    ));
    if pending.exists() {
        return Err("preserving incomplete station rows".into());
    }
    let _cleanup = PendingOutput(pending.clone());
    let mut out = csv::Writer::from_writer(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&pending)?,
    );
    out.write_record([
        "date",
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "distance_km",
        "product",
        "forecast",
        "station",
        "error",
        "station_options",
        "calcstatus",
    ])?;
    let opts = col("opts")?;
    let mut stats: BTreeMap<String, Stats> = BTreeMap::new();
    let mut stations = BTreeSet::new();
    let mut seen = BTreeMap::new();
    let mut identical_duplicates = 0usize;
    let mut ambiguous = BTreeSet::new();
    let mut candidates = Vec::new();
    let mut scanned = 0usize;
    let mut matched = 0usize;
    let start_date = r.start_date.to_string();
    let end_date = r.end_date.to_string();
    for rec in reader.records() {
        let rec = rec?;
        scanned += 1;
        let ds = rec.get(date).unwrap_or("");
        let Some(ds) = ds.get(..10) else { continue };
        if ds < start_date.as_str() || ds > end_date.as_str() {
            continue;
        }
        let d = NaiveDate::parse_from_str(ds, "%Y-%m-%d")?;
        if !days.contains_key(&d) {
            continue;
        }
        // calcstatus=1 is CWFIS active FWI calculation; nonpositive codes are inactive.
        if rec[status].parse::<i32>().unwrap_or(0) != 1 {
            continue;
        }
        let Ok(sy) = rec[y].parse::<f64>() else {
            continue;
        };
        let Ok(sx) = rec[x].parse::<f64>() else {
            continue;
        };
        if !sy.is_finite() || !sx.is_finite() {
            continue;
        }
        let near = |a: &[f64], v: f64| {
            a.iter()
                .enumerate()
                .min_by(|(_, x), (_, y)| (*x - v).abs().total_cmp(&(*y - v).abs()))
                .map(|(i, _)| i)
                .unwrap()
        };
        let iy = near(&lat, sy);
        let ix = near(&lon, sx);
        let dy = (lat[iy] - sy).to_radians();
        let dx = (lon[ix] - sx).to_radians();
        let a = (dy / 2.).sin().powi(2)
            + sy.to_radians().cos() * lat[iy].to_radians().cos() * (dx / 2.).sin().powi(2);
        let distance = 6371. * 2. * a.sqrt().min(1.).asin();
        if distance > r.max_distance_km {
            continue;
        }
        // Short local station identifiers are reused by different networks.
        // Preserve site identity as well as the identifier before deduplicating.
        let site = (
            rec[id].to_string(),
            rec[name].to_string(),
            sy.to_bits(),
            sx.to_bits(),
        );
        let key = (d, site.clone());
        if let Some(previous) = seen.get(&key) {
            if previous == &rec {
                identical_duplicates += 1;
            } else {
                ambiguous.insert(key);
            }
            continue;
        }
        seen.insert(key, rec.clone());
        candidates.push((d, site, rec, sy, sx, distance, iy, ix));
    }
    // An archive can contain conflicting recalculations for one site/day. No
    // revision ordering is documented: quarantine the entire ambiguous group,
    // rather than choosing whichever record happens to appear first.
    for (d, site, rec, sy, sx, distance, iy, ix) in candidates {
        if ambiguous.contains(&(d, site.clone())) {
            continue;
        }
        let fields = &days[&d];
        let date_string = d.to_string();
        let ds = date_string.as_str();
        let cell = iy * lon.len() + ix;
        let mut used = false;
        for (p, &c) in obs.iter().enumerate() {
            let Ok(o) = rec[c].parse::<f64>() else {
                continue;
            };
            let f = fields[p][cell];
            if !o.is_finite() || o < 0. || !f.is_finite() {
                continue;
            }
            let product = PRODUCTS[p];
            stats.entry(product.into()).or_default().add(f, o);
            stats.entry(format!("{d}/{product}")).or_default().add(f, o);
            used = true;
            out.write_record([
                ds,
                &rec[id],
                &rec[name],
                &sy.to_string(),
                &sx.to_string(),
                &distance.to_string(),
                product,
                &f.to_string(),
                &o.to_string(),
                &(f - o).to_string(),
                &rec[opts],
                &rec[status],
            ])?;
        }
        if used {
            matched += 1;
            stations.insert(site);
        }
    }
    out.flush()?;
    if matched == 0 {
        return Err("no station/forecast pairs within requested dates/distance".into());
    }
    drop(out);
    fs::rename(&pending, &r.rows)?;
    let metrics: BTreeMap<_, _> = stats.iter().map(|(k, v)| (k, v.value())).collect();
    publish(
        &r.receipt,
        &serde_json::to_vec_pretty(
            &json!({"schema":"arwen.fire-station-verification.v1","archive":r.archive,"archive_sha256":hash(&r.archive)?,"license":r.research_station_license,"source":"Canadian Forest Service, CWFIS station FWI archive","forecasts":sources,"rows":r.rows,"rows_sha256":hash(&r.rows)?,"scanned":scanned,"station_days":matched,"stations":stations.len(),"identical_archive_duplicates":identical_duplicates,"ambiguous_station_days_excluded":ambiguous.len(),"duplicate_policy":"all differing records at the same physical site/day excluded; identical duplicates collapsed","max_distance_km":r.max_distance_km,"wall_seconds":wall.elapsed().as_secs_f64(),"stage_wall_seconds":{"decompress":decode_wall},"parallelism":{"decompress":1,"station_match":1},"metrics":metrics,"limitations":["comparison uses indices derived from observed station weather; these are not observed fire behaviour","nearest grid point, no elevation correction","FWI startup/overwintering differences remain in drought codes","KBDI and NFDRS have no verification rows here"]}),
        )?,
    )?;
    Ok(())
}
