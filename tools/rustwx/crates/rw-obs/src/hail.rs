//! Hail observations and size-bin verification. SPC is a sparse positive-report
//! sample; missing reports never count as no hail. MESH is a radar estimate,
//! not a ground measurement. Sources and every input digest ride the receipt.
use crate::pack::{ArrayEntry, decode_pack, payload_digest, validate_arrays};
use crate::{err, hex_sha256};
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, TimeZone, Utc};
use obs_regrid::{Method, apply_plan, build_plan};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    error::Error,
    path::{Path, PathBuf},
};
use wrf_core::WrfFile;
#[path = "csv.rs"]
mod csv;
type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub const ABI: &str = "gpuwm-obs.hail.v1\tSPC\tMESH\tmm\tsize_bins";
pub const BINS_MM: &[f64] = &[0., 25.4, 38.1, 50.8, 76.2, 101.6, 152.4, 203.2];
pub const FIELDS: &[(&str, f64)] = &[
    ("HAIL_MAXK1", 1000.),
    ("HAIL_MAX2D", 1000.),
    ("HAILCAST_DIAM_MAX", 1.),
];
fn bin(x: f64) -> usize {
    BINS_MM.iter().rposition(|b| x >= *b).unwrap_or(0)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Report {
    pub time: DateTime<Utc>,
    pub latitude: f64,
    pub longitude: f64,
    pub size_mm: f64,
}
pub fn parse_spc(text: &str, day: &str) -> Result<Vec<Report>> {
    if day.len() != 8 || !day.bytes().all(|b| b.is_ascii_digit()) {
        return Err(err("SPC day must be YYYYMMDD"));
    }
    let date = NaiveDate::parse_from_str(day, "%Y%m%d")?;
    let rows = csv::parse(text).map_err(err)?;
    let header = rows.first().ok_or_else(|| err("SPC CSV is empty"))?;
    for (i, name) in [(0, "Time"), (1, "Size"), (5, "Lat"), (6, "Lon")] {
        if header.get(i).map(String::as_str) != Some(name) {
            return Err(err("SPC hail CSV header differs"));
        }
    }
    let mut reports = Vec::new();
    for row in rows
        .iter()
        .skip(1)
        .filter(|r| r.iter().any(|v| !v.trim().is_empty()))
    {
        if row.len() < 7 {
            return Err(err("SPC hail CSV has a truncated row"));
        }
        if row[1].trim() == "UNK" {
            continue;
        }
        let clock = row[0].parse::<u32>()?;
        let hour = clock / 100;
        let minute = clock % 100;
        let d = if hour < 12 {
            date + Duration::days(1)
        } else {
            date
        };
        let time = Utc.from_utc_datetime(
            &d.and_hms_opt(hour, minute, 0)
                .ok_or_else(|| err("invalid SPC UTC clock"))?,
        );
        let size_mm = row[1].parse::<f64>()? * 0.254;
        let latitude = row[5].parse::<f64>()?;
        let longitude = row[6].parse::<f64>()?;
        if !size_mm.is_finite()
            || size_mm <= 0.
            || size_mm > 1000.
            || !(-90.0..=90.0).contains(&latitude)
            || !(-180.0..=180.0).contains(&longitude)
        {
            return Err(err("SPC row has invalid coordinates or hail size"));
        }
        reports.push(Report {
            time,
            latitude,
            longitude,
            size_mm,
        });
    }
    Ok(reports)
}
pub fn spc(day: &str, file: Option<&Path>, out: &Path) -> Result<Value> {
    if day.len() != 8 || !day.bytes().all(|b| b.is_ascii_digit()) {
        return Err(err("SPC day must be YYYYMMDD"));
    }
    NaiveDate::parse_from_str(day, "%Y%m%d")?;
    let url = format!(
        "https://www.spc.noaa.gov/climo/reports/{}_rpts_hail.csv",
        &day[2..]
    );
    let raw = if let Some(p) = file {
        std::fs::read(p)?
    } else {
        crate::net::get_bytes(&crate::net::agent(), &url, "SPC hail reports")?
    };
    let reports = parse_spc(std::str::from_utf8(&raw)?, day)?;
    let record = json!({"schema":"gpuwm-obs.spc-hail.v1","status":"READY","source":url,
  "source_sha256":hex_sha256(&raw),"day":day,"units":"mm","reports":reports,
  "interpretation":"Preliminary sparse positive reports; absence is not a negative observation",
  "size_bins_mm":BINS_MM});
    std::fs::write(out, serde_json::to_vec_pretty(&record)?)?;
    Ok(record)
}
#[derive(Deserialize)]
pub struct Frame {
    pub wrfout: PathBuf,
    pub valid_time: DateTime<Utc>,
    pub window_seconds: i64,
    pub mesh: PathBuf,
    pub geometry: PathBuf,
}
#[derive(Deserialize)]
pub struct Request {
    pub schema: String,
    pub frames: Vec<Frame>,
    pub spc: PathBuf,
    pub max_distance_m: f64,
}
fn pack(path: &Path, schema: &str) -> Result<(Value, Vec<u8>, BTreeMap<String, ArrayEntry>)> {
    let (m, p): (Value, Vec<u8>) = decode_pack(&std::fs::read(path)?)?;
    if m["schema"] != schema
        || m["status"] != "READY"
        || m["content_sha256"] != payload_digest(&p)
        || m["payload_bytes"].as_u64() != Some(p.len() as u64)
    {
        return Err(err(format!(
            "unready or corrupt observation pack: {}",
            path.display()
        )));
    }
    let arrays: BTreeMap<String, ArrayEntry> = serde_json::from_value(m["arrays"].clone())?;
    validate_arrays(&arrays, p.len())?;
    Ok((m, p, arrays))
}
fn f64s(
    p: &[u8],
    arrays: &BTreeMap<String, ArrayEntry>,
    key: &str,
    shape: &[usize],
) -> Result<Vec<f64>> {
    let a = arrays
        .get(key)
        .ok_or_else(|| err(format!("pack missing {key}")))?;
    if a.dtype != "<f8" || a.shape != shape {
        return Err(err(format!("{key} has wrong units/layout")));
    }
    Ok(p[a.offset..a.offset + a.bytes]
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn plane(f: &WrfFile, name: &str) -> Result<Vec<f64>> {
    let v = f.read_var(name, 0)?;
    if v.len() != f.nxy() {
        return Err(err(format!(
            "{name} has {} values, expected {} mass-grid cells",
            v.len(),
            f.nxy()
        )));
    }
    if let Some(first) = v.iter().position(|x| !x.is_finite()) {
        let count = v.iter().filter(|x| !x.is_finite()).count();
        return Err(err(format!(
            "{name} has {count} nonfinite cells; first at (i={}, j={})",
            first % f.nx,
            first / f.nx
        )));
    }
    Ok(v)
}
pub fn verify(req: &Request) -> Result<Value> {
    if req.schema != "gpuwm.hail-verification-request.v1"
        || req.frames.is_empty()
        || !req.max_distance_m.is_finite()
        || req.max_distance_m <= 0.
        || req.max_distance_m > 10000.
    {
        return Err(err(
            "hail verification needs frames and a remap bound within 10 km",
        ));
    }
    let spc: Value = serde_json::from_slice(&std::fs::read(&req.spc)?)?;
    if spc["schema"] != "gpuwm-obs.spc-hail.v1" || spc["status"] != "READY" || spc["units"] != "mm"
    {
        return Err(err("SPC input is not a ready mm hail report record"));
    }
    let reports: Vec<Report> = serde_json::from_value(spc["reports"].clone())?;
    let mut frames = Vec::new();
    let mut previous = None;
    if reports.iter().any(|r| {
        !r.size_mm.is_finite()
            || r.size_mm <= 0.
            || r.size_mm > 1000.
            || !(-90.0..=90.0).contains(&r.latitude)
            || !(-180.0..=180.0).contains(&r.longitude)
    }) {
        return Err(err("SPC record contains invalid size or coordinates"));
    }
    for frame in &req.frames {
        if frame.window_seconds != 1800
            || previous.is_some_and(|v| frame.valid_time - v != Duration::seconds(1800))
        {
            return Err(err(
                "verification requires contiguous 30-minute forecast windows against MESH_Max_30min",
            ));
        }
        previous = Some(frame.valid_time);
        let file = WrfFile::open(&frame.wrfout)?;
        let times = file.times()?;
        let time = times.first().ok_or_else(|| err("WRF Times is empty"))?;
        if file.nt != 1
            || Utc.from_utc_datetime(&NaiveDateTime::parse_from_str(time, "%Y-%m-%d_%H:%M:%S")?)
                != frame.valid_time
        {
            return Err(err(
                "forecast Times does not match the requested verification window end",
            ));
        }
        let metadata = netcdf_reader::NcFile::open(&frame.wrfout)?;
        let lat = plane(&file, "XLAT")?;
        let lon = plane(&file, "XLONG")?;
        let shape = (file.ny, file.nx);
        let cells = file.nxy();
        let (m, p, arrays) = pack(&frame.mesh, "gpuwm-obs.obs-grid.v1")?;
        if m["quantity"] != "maximum_estimated_hail_size"
            || m["units"] != "mm"
            || m["provenance"]["is_stub"] != false
            || m["provenance"]["product"] != "MESH_Max_30min_00.50"
        {
            return Err(err(
                "verification needs observed MESH_Max_30min in mm, not reflectivity or instantaneous MESH",
            ));
        }
        let mt = Utc.from_utc_datetime(&NaiveDateTime::parse_from_str(
            m["valid_time"]
                .as_str()
                .ok_or_else(|| err("MESH time missing"))?,
            crate::seam::TIME_FORMAT,
        )?);
        if (mt - frame.valid_time).num_seconds().abs() > 120 {
            return Err(err(
                "MESH and forecast window ends differ by more than 120 seconds",
            ));
        }
        let (g, gp, ga) = pack(&frame.geometry, "gpuwm-obs.obs-geo.v1")?;
        if m["grid"] != g["grid"] || g["source_product"] != "MESH_Max_30min_00.50" {
            return Err(err("MESH geometry/product differs"));
        }
        let os = [
            m["grid"]["ny"]
                .as_u64()
                .ok_or_else(|| err("MESH ny missing"))? as usize,
            m["grid"]["nx"]
                .as_u64()
                .ok_or_else(|| err("MESH nx missing"))? as usize,
        ];
        let olat = f64s(&gp, &ga, "latitude", &os)?;
        let olon = f64s(&gp, &ga, "longitude", &os)?;
        let observed = f64s(&p, &arrays, "values", &os)?;
        let a = arrays
            .get("valid")
            .ok_or_else(|| err("MESH validity missing"))?;
        if a.dtype != "<u1" || a.shape != os {
            return Err(err("MESH validity layout differs"));
        }
        let valid = p[a.offset..a.offset + a.bytes]
            .iter()
            .map(|x| *x != 0)
            .collect::<Vec<_>>();
        if observed
            .iter()
            .zip(&valid)
            .any(|(v, m)| *m && (!v.is_finite() || *v < 0. || *v > 1000.))
        {
            return Err(err("MESH valid diameter outside [0,1000] mm"));
        }
        let plan = build_plan(
            Method::Nearest,
            &olat,
            &olon,
            (os[0], os[1]),
            &lat,
            &lon,
            shape,
            req.max_distance_m,
        )?;
        let mut obs = vec![0.; cells];
        let mut mask = vec![false; cells];
        apply_plan(
            Method::Nearest,
            &plan.source_index,
            &plan.reachable,
            plan.source_shape,
            plan.destination_shape,
            &observed,
            &valid,
            &mut obs,
            &mut mask,
        )?;
        // Five boundary rows are lateral forcing, not independently forecast skill.
        let mut interior = vec![true; cells];
        for y in 0..file.ny {
            for x in 0..file.nx {
                if x < 5 || y < 5 || x + 5 >= file.nx || y + 5 >= file.ny {
                    mask[y * file.nx + x] = false;
                    interior[y * file.nx + x] = false;
                }
            }
        }
        let model_fields=FIELDS.iter().map(|(name,scale)| {
   let expected=if *scale==1000. {"m"}else{"mm"};
   let variable=metadata.variable(name)?;
   let units=variable.attributes.iter().find(|a|a.name=="units").and_then(|a|a.value.as_string());
   if units.as_deref()!=Some(expected) {return Err(err(format!("{name} must declare units {expected}")));}
   let mut model=plane(&file,name).map_err(|e| err(format!("{}: {e}", frame.wrfout.display())))?;model.par_iter_mut().for_each(|v|*v *= scale);
   if model.iter().any(|x|*x<0.||*x>1000.) {return Err(err("model hail diameter outside [0,1000] mm"));}
   let confusion=(0..cells).into_par_iter().fold(||vec![0u64;64],|mut c,i|{if mask[i]{c[bin(obs[i])*8+bin(model[i])]+=1;}c}).reduce(||vec![0u64;64],|mut a,b|{for(i,n)in b.iter().enumerate(){a[i]+=n;}a});
   let counts=mask.iter().filter(|x|**x).count();if counts==0{return Err(err("MESH has no matched interior cells"));}
   let squared=(0..cells).into_par_iter().filter(|i|mask[*i]).map(|i|(model[i]-obs[i]).powi(2)).sum::<f64>();
   let absolute=(0..cells).into_par_iter().filter(|i|mask[*i]).map(|i|(model[i]-obs[i]).abs()).sum::<f64>();
   let thresholds=[25.4,38.1,50.8,76.2].iter().map(|threshold| {
    let counts=(0..cells).into_par_iter().fold(||[0u64;4],|mut c,i| {
     if mask[i] {let o=obs[i]>=*threshold;let f=model[i]>=*threshold;c[match(o,f){(true,true)=>0,(false,true)=>1,(true,false)=>2,_=>3}]+=1;}c
    }).reduce(||[0;4],|mut a,b|{for i in 0..4{a[i]+=b[i];}a});
    let [hit,false_alarm,miss,correct_negative]=counts;
    let ratio=|n:u64,d:u64|if d>0{Some(n as f64/d as f64)}else{None};
    json!({"threshold_mm":threshold,"hits":hit,"false_alarms":false_alarm,"misses":miss,"correct_negatives":correct_negative,
     "pod":ratio(hit,hit+miss),"far":ratio(false_alarm,hit+false_alarm),"csi":ratio(hit,hit+false_alarm+miss)})
   }).collect::<Vec<_>>();
   let points=reports.par_iter().filter(|r|r.time>frame.valid_time-Duration::seconds(1800)&&r.time<=frame.valid_time).filter_map(|r| {
    let distances=lat.iter().zip(&lon).enumerate().filter(|(i,_)|interior[*i]).map(|(i,(y,x))|{let dy=(y-r.latitude).to_radians();let dx=(x-r.longitude).to_radians()*r.latitude.to_radians().cos();(i,6371000.*(dx*dx+dy*dy).sqrt())}).collect::<Vec<_>>();
    let (nearest,dist)=distances.iter().min_by(|a,b|a.1.total_cmp(&b.1)).copied()?;
    if dist>req.max_distance_m{return None;}
    let neighborhood=distances.iter().filter(|(_,d)|*d<=40000.).map(|(i,_)|model[*i]).fold(0.,f64::max);
    Some(json!({"report":r,"nearest_distance_m":dist,"model_mm":model[nearest],"model_bin":bin(model[nearest]),"observed_bin":bin(r.size_mm),"neighborhood_40km_max_mm":neighborhood}))
   }).collect::<Vec<_>>();
   Ok(json!({"field":name,"units":"mm","matched_cells":counts,"mae_mm":absolute/counts as f64,"rmse_mm":(squared/counts as f64).sqrt(),"confusion_observed_rows_model_columns":confusion,"threshold_scores":thresholds,"spc_positive_matches":points}))
  }).collect::<Result<Vec<_>>>()?;
        frames.push(json!({"spc_reports_in_time_window":reports.iter().filter(|r|r.time>frame.valid_time-Duration::seconds(1800)&&r.time<=frame.valid_time).count(),"valid_time":frame.valid_time,"window_seconds":1800,"fields":model_fields,
   "wrfout":frame.wrfout,"wrfout_sha256":hex_sha256(&std::fs::read(&frame.wrfout)?),
   "mesh":frame.mesh,"mesh_sha256":hex_sha256(&std::fs::read(&frame.mesh)?),"geometry":frame.geometry,"geometry_sha256":hex_sha256(&std::fs::read(&frame.geometry)?)}));
    }
    Ok(
        json!({"schema":"gpuwm.hail-verification.v1","status":"READY","size_bin_lower_bounds_mm":BINS_MM,
  "mesh_interpretation":"Radar estimated size, not measured stones","spc_interpretation":"Sparse positive reports only; no false-alarm inference from absent reports",
  "remap":"nearest MESH grid centre onto model; validity carried; 5 boundary rows excluded","max_distance_m":req.max_distance_m,"spc_sha256":hex_sha256(&std::fs::read(&req.spc)?),"frames":frames}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spc_rolls_utc_at_noon_and_converts_hundredths() {
        let r=parse_spc("Time,Size,Location,County,State,Lat,Lon,Comments\n2359,175,X,X,IA,41,-92,\"hail, large\"\n0015,250,X,X,IA,41,-92,X\n","20230331").unwrap();
        assert_eq!(r[0].time.to_rfc3339(), "2023-03-31T23:59:00+00:00");
        assert_eq!(r[1].time.to_rfc3339(), "2023-04-01T00:15:00+00:00");
        assert!((r[0].size_mm - 44.45).abs() < 1e-9);
    }
    #[test]
    fn size_bins_include_the_lower_threshold() {
        assert_eq!(bin(25.399), 0);
        assert_eq!(bin(25.4), 1);
        assert_eq!(bin(50.8), 3);
    }
    #[test]
    fn bad_spc_values_refuse() {
        assert!(
            parse_spc(
                "Time,Size,X,X,X,Lat,Lon\n2500,100,X,X,X,41,-92\n",
                "20230331"
            )
            .is_err()
        );
    }
}
