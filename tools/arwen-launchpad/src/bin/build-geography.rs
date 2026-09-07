//! Offline display-asset builder. This never touches forecast/WPS geography.
//! Inputs: existing Natural Earth10m SHP layers and a pinned GeoNames extract.
use serde_json::{json, Value};
use std::{collections::HashMap, fmt::Write, fs, path::Path};

const TOLERANCE: f64 = 0.0025; // degrees, display simplification only
fn u32le(b: &[u8], p: usize) -> usize {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize
}
fn f64le(b: &[u8], p: usize) -> f64 {
    f64::from_le_bytes(b[p..p + 8].try_into().unwrap())
}
fn distance2(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let norm = dx * dx + dy * dy;
    let t = if norm == 0.0 {
        0.0
    } else {
        ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / norm
    }
    .clamp(0.0, 1.0);
    (p[0] - a[0] - t * dx).powi(2) + (p[1] - a[1] - t * dy).powi(2)
}
fn simplify(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0, points.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        let mut far = TOLERANCE * TOLERANCE;
        let mut index = None;
        for i in a + 1..b {
            let d = distance2(points[i], points[a], points[b]);
            if d > far {
                far = d;
                index = Some(i);
            }
        }
        if let Some(i) = index {
            keep[i] = true;
            stack.push((a, i));
            stack.push((i, b));
        }
    }
    points
        .iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(*p))
        .collect()
}
fn layer(file: &Path, polygon: bool) -> Result<Vec<String>, String> {
    let bytes = fs::read(file).map_err(|e| e.to_string())?;
    if bytes.len() < 100 || u32::from_be_bytes(bytes[..4].try_into().unwrap()) != 9994 {
        return Err(format!("Invalid SHP header: {}", file.display()));
    }
    let mut offset = 100;
    let mut paths = vec![];
    while offset < bytes.len() {
        if offset + 8 > bytes.len() {
            return Err("Truncated SHP record header".into());
        }
        let size =
            (u32::from_be_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize) * 2;
        offset += 8;
        let record = bytes
            .get(offset..offset + size)
            .ok_or("Truncated SHP record")?;
        offset += size;
        if record.len() < 4 {
            return Err("Missing SHP shape type".into());
        }
        let kind = u32le(record, 0);
        if kind == 0 {
            continue;
        }
        if kind != if polygon { 5 } else { 3 } || record.len() < 44 {
            return Err(format!("Unsupported/truncated SHP shape{kind}"));
        }
        let nparts = u32le(record, 36);
        let npoints = u32le(record, 40);
        let point_start = 44usize
            .checked_add(nparts.checked_mul(4).ok_or("Part count overflow")?)
            .ok_or("Part overflow")?;
        let end = point_start
            .checked_add(npoints.checked_mul(16).ok_or("Point count overflow")?)
            .ok_or("Point overflow")?;
        if end > record.len() {
            return Err("Truncated SHP points".into());
        }
        let mut d = String::new();
        for part in 0..nparts {
            let a = u32le(record, 44 + part * 4);
            let b = if part + 1 < nparts {
                u32le(record, 44 + (part + 1) * 4)
            } else {
                npoints
            };
            if a >= b || b > npoints {
                return Err("Invalid SHP part index".into());
            }
            let points: Vec<_> = (a..b)
                .map(|i| {
                    [
                        f64le(record, point_start + i * 16),
                        f64le(record, point_start + i * 16 + 8),
                    ]
                })
                .collect();
            if points.iter().any(|p| {
                !p[0].is_finite()
                    || !p[1].is_finite()
                    || p[0].abs() > 180.001
                    || p[1].abs() > 90.001
            }) {
                return Err("SHP is not finite longitude/latitude".into());
            }
            let points = simplify(&points);
            if points.len() < if polygon { 4 } else { 2 } {
                continue;
            }
            for (i, p) in points.iter().enumerate() {
                write!(
                    &mut d,
                    "{}{:.3},{:.3}",
                    if i == 0 { 'M' } else { 'L' },
                    (p[0] + 180.0) * 2.0,
                    (90.0 - p[1]) * 2.0
                )
                .unwrap();
            }
            if polygon {
                d.push('Z');
            }
        }
        if !d.is_empty() {
            paths.push(d);
        }
    }
    Ok(paths)
}
fn places(source: &Path) -> Result<Vec<Value>, String> {
    let admin =
        fs::read_to_string(source.join("admin1CodesASCII.txt")).map_err(|e| e.to_string())?;
    let admin: HashMap<_, _> = admin
        .lines()
        .filter_map(|l| {
            let p: Vec<_> = l.split('\t').collect();
            (p.len() >= 4).then(|| (p[0].to_string(), p[1].to_string()))
        })
        .collect();
    let countries =
        fs::read_to_string(source.join("countryInfo.txt")).map_err(|e| e.to_string())?;
    let countries: HashMap<_, _> = countries
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let p: Vec<_> = l.split('\t').collect();
            (p.len() > 4).then(|| (p[0].to_string(), p[4].to_string()))
        })
        .collect();
    let mut result = vec![];
    for line in fs::read_to_string(source.join("cities15000.txt"))
        .map_err(|e| e.to_string())?
        .lines()
    {
        let p: Vec<_> = line.split('\t').collect();
        if p.len() != 19 {
            return Err("GeoNames row must contain19 fields".into());
        }
        let lat: f64 = p[4].parse().map_err(|_| "Invalid latitude")?;
        let lon: f64 = p[5].parse().map_err(|_| "Invalid longitude")?;
        if !lat.is_finite() || !lon.is_finite() || lat.abs() > 90.0 || lon.abs() > 180.0 {
            return Err("Invalid place coordinates".into());
        }
        let id: u64 = p[0].parse().map_err(|_| "Invalid place ID")?;
        let population: u64 = p[14].parse().map_err(|_| "Invalid place population")?;
        let region = admin
            .get(&format!("{}.{}", p[8], p[10]))
            .map(String::as_str)
            .unwrap_or(p[10]);
        let country = countries.get(p[8]).map(String::as_str).unwrap_or(p[8]);
        result.push(json!([
            id, p[1], p[2], region, country, p[8], lat, lon, population, p[3]
        ]));
    }
    result.sort_by_key(|p| std::cmp::Reverse(p[8].as_u64().unwrap()));
    Ok(result)
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("build-geography BASEMAP_DIR GEONAMES_DIR OUTPUT_JSON".into());
    }
    let base = Path::new(&args[1]).join("natural_earth_10m");
    let source = Path::new(&args[2]);
    let data = json!({"schema":"arwen.offline-geography.v1","display_only":true,"simplification_degrees":TOLERANCE,
        "land":layer(&base.join("ne_10m_land.shp"),true)?,"lakes":layer(&base.join("ne_10m_lakes.shp"),true)?,
        "boundaries":layer(&base.join("ne_10m_admin_1_states_provinces_lines.shp"),false)?,
        "countries":layer(&base.join("ne_10m_admin_0_boundary_lines_land.shp"),false)?,"places":places(source)?,
        "place_columns":["id","name","ascii_name","region","country","country_code","latitude","longitude","population","alternate_names"]});
    let body = serde_json::to_vec(&data).map_err(|e| e.to_string())?;
    println!(
        "{} places; {} land, {} lake, {} province, {} country paths; {} bytes",
        data["places"].as_array().unwrap().len(),
        data["land"].as_array().unwrap().len(),
        data["lakes"].as_array().unwrap().len(),
        data["boundaries"].as_array().unwrap().len(),
        data["countries"].as_array().unwrap().len(),
        body.len()
    );
    fs::write(&args[3], body).map_err(|e| e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simplification_retains_bends_and_endpoint_coordinates() {
        let p = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]];
        assert_eq!(simplify(&p), vec![[0.0, 0.0], [2.0, 0.0], [2.0, 1.0]]);
    }
    #[test]
    fn closed_ring_does_not_collapse_into_a_line() {
        let p = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
        assert_eq!(simplify(&p), p);
    }
}
