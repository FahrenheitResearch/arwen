//! Public GLM flash-count verification of native lightning diagnostics.
//! File coverage is mandatory and contiguous; missing files are never dry cells.
use crate::verification_io::{self, ArmSpec};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::PathBuf;
pub const REQUEST_SCHEMA: &str = "gpuwm.lightning-verification.request.v1";
pub const RECEIPT_SCHEMA: &str = "gpuwm.lightning-verification.receipt.v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema: String,
    pub valid_time: String,
    pub window_start: String,
    pub window_end: String,
    pub arm: ArmSpec,
    /// Native grid's invariant file: MAPFAC_M, XLAT, XLONG, DX and DY.
    pub area_source_path: PathBuf,
    pub glm_paths: Vec<PathBuf>,
    /// Observed footprint selected from provider coverage, [west,south,east,north].
    pub bbox: [f64; 4],
}
const PRODUCTS: &[(&str, bool)] = &[
    ("lightning_threat_1", true),
    ("lightning_threat_2", true),
    ("lightning_threat_3", true),
    ("pr92_flash_rate", false),
];
fn read(f: &netcrust::File, name: &str) -> Result<Vec<f64>, String> {
    f.variable(name)
        .ok_or_else(|| format!("missing GLM/grid variable {name}"))?
        .values_f64()
        .map_err(|e| e.to_string())
}
fn attr(f: &netcrust::File, name: &str) -> Result<String, String> {
    f.attribute(name)
        .and_then(|a| a.as_string().map(str::to_owned))
        .ok_or_else(|| format!("missing GLM attribute {name}"))
}
/// Continuous counts, with forecast minus observation error on common cells.
fn count_score(model: &[f64], observed: &[u64]) -> Result<Value, String> {
    if model.len() != observed.len() || model.is_empty() {
        return Err("count grid shape mismatch".into());
    }
    let rows: Vec<_> = model
        .par_iter()
        .zip(observed)
        .filter_map(|(m, o)| m.is_finite().then_some((*m, *o as f64)))
        .collect();
    if rows.is_empty() {
        return Err("no finite common lightning coverage".into());
    }
    let (fm, ob, ae, se) = rows
        .par_iter()
        .map(|(m, o)| (*m, *o, (m - o).abs(), (m - o).powi(2)))
        .reduce(
            || (0., 0., 0., 0.),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
        );
    let n = rows.len() as f64;
    Ok(
        json!({"common_cells":rows.len(),"excluded_or_missing_cells":model.len()-rows.len(),"forecast_flashes":fm,"observed_flashes":ob,"bias_flashes_per_cell":(fm-ob)/n,"mae_flashes_per_cell":ae/n,"rmse_flashes_per_cell":(se/n).sqrt(),"observed_active_cells":rows.par_iter().filter(|(_,o)|*o>0.).count()}),
    )
}
pub fn score(mut request: Request) -> Result<Value, String> {
    if request.schema != REQUEST_SCHEMA {
        return Err("unsupported lightning verification schema".into());
    }
    let [west, south, east, north] = request.bbox;
    if request.bbox.iter().any(|v| !v.is_finite())
        || west >= east
        || south >= north
        || west < -180.
        || east > 180.
        || south < -90.
        || north > 90.
    {
        return Err("invalid observed coverage bbox".into());
    }
    if request.arm.kind != "store" {
        return Err("lightning verification requires the native diagnostic store".into());
    }
    let start = crate::verification::parse_time(&request.window_start)?;
    let end = crate::verification::parse_time(&request.window_end)?;
    if end - start != chrono::Duration::seconds(300)
        || crate::verification::parse_time(&request.valid_time)? != start
    {
        return Err(
            "LTG count comparison requires a 300 s window starting at model valid time".into(),
        );
    }
    if request.glm_paths.is_empty() {
        return Err("no GLM files supplied".into());
    }
    if request.arm.fields.is_empty() {
        request.arm.fields = PRODUCTS
            .iter()
            .map(|(n, _)| (n.to_string(), n.to_string()))
            .collect()
    }
    let arm = verification_io::load_arm(&request.arm)?;
    verification_io::validate_arm_time(&request.arm, &arm, &request.valid_time)?;
    let grid = arm
        .grid
        .as_ref()
        .ok_or("lightning verification needs a native grid")?;
    let cells = grid.nx * grid.ny;
    let eligible: Vec<bool> = (0..cells)
        .into_par_iter()
        .map(|i| {
            f64::from(grid.lat[i]) >= south
                && f64::from(grid.lat[i]) <= north
                && f64::from(grid.lon[i]) >= west
                && f64::from(grid.lon[i]) <= east
        })
        .collect();
    if !eligible.iter().any(|x| *x) {
        return Err("observed coverage bbox has no native cells".into());
    }
    let reader =
        rw_store::reader::HourReader::open(&request.arm.path).map_err(|e| e.to_string())?;
    for &(name, density) in PRODUCTS {
        let key = request
            .arm
            .fields
            .get(name)
            .ok_or_else(|| format!("unmapped diagnostic {name}"))?;
        let units = &reader
            .variable(key)
            .ok_or_else(|| format!("missing stored diagnostic {key}"))?
            .units;
        let accepted = if density {
            ["flashes km^-2 (5 min)^-1", "flash km^-2 (5 min)^-1"]
        } else {
            ["flashes s^-1 cell^-1", "flash s^-1 cell^-1"]
        };
        if !accepted.contains(&units.as_str()) {
            return Err(format!("{name} has incompatible units {units}"));
        }
    }
    let source = netcrust::open(&request.area_source_path).map_err(|e| e.to_string())?;
    let dx = source
        .attribute("DX")
        .and_then(|a| a.as_f64())
        .ok_or("area source DX missing")?;
    let dy = source
        .attribute("DY")
        .and_then(|a| a.as_f64())
        .ok_or("area source DY missing")?;
    let mf = read(&source, "MAPFAC_M")?;
    let lat = read(&source, "XLAT")?;
    let lon = read(&source, "XLONG")?;
    if [mf.len(), lat.len(), lon.len()].iter().any(|n| *n != cells)
        || dx <= 0.
        || dy <= 0.
        || !dx.is_finite()
        || !dy.is_finite()
    {
        return Err("area source dimensions/spacing invalid".into());
    }
    if (0..cells).into_par_iter().any(|i| {
        !mf[i].is_finite()
            || mf[i] <= 0.
            || !lat[i].is_finite()
            || !lon[i].is_finite()
            || (lat[i] - f64::from(grid.lat[i])).abs() > 1e-4
            || (lon[i] - f64::from(grid.lon[i])).abs() > 1e-4
    }) {
        return Err("area source and forecast grid differ".into());
    }
    let areas: Vec<f64> = mf.par_iter().map(|m| dx * dy / (1e6 * m * m)).collect();
    let mut files=request.glm_paths.par_iter().map(|p|{
        let file=netcrust::open(p).map_err(|e|e.to_string())?;
        let s=crate::verification::parse_time(&attr(&file,"time_coverage_start")?)?;
        let e=crate::verification::parse_time(&attr(&file,"time_coverage_end")?)?;
        if e<=s{return Err("nonpositive GLM coverage".to_string())}
        let lat=read(&file,"flash_lat")?;let lon=read(&file,"flash_lon")?;let q=read(&file,"flash_quality_flag")?;
        if lat.len()!=lon.len()||lat.len()!=q.len(){return Err("GLM flash arrays disagree".into())}
        let mut indices=Vec::new();let mut rejected=0usize;let mut outside=0usize;
        for ((lat,lon),q) in lat.iter().zip(&lon).zip(&q){
            if *q!=0.||!lat.is_finite()||!lon.is_finite()||lat.abs()>90.||lon.abs()>180.{rejected+=1;continue}
            if let Some((x,y))=grid.position(*lat,*lon){
                let i=(y.round() as usize).min(grid.ny-1)*grid.nx+(x.round() as usize).min(grid.nx-1);
                if eligible[i]{indices.push(i)}else{outside+=1}
            }else{outside+=1}
        }
        let provenance=json!({"artifact":verification_io::source_identity(p)?,"start":s.to_rfc3339(),"end":e.to_rfc3339(),"quality_rejected":rejected,"outside_native_grid":outside,"legacy_file_license":file.attribute("license").and_then(|a|a.as_string().map(str::to_owned))});
        Ok((s,e,indices,provenance))
    }).collect::<Result<Vec<_>,String>>()?;
    files.sort_by_key(|f| f.0);
    let mut cursor = start;
    let mut observed = vec![0u64; cells];
    let mut provenance = Vec::new();
    for (s, e, indices, identity) in files {
        if s != cursor || e > end {
            return Err(format!(
                "GLM coverage gap, overlap or out-of-window file: expected {cursor}, got {s}..{e}"
            ));
        }
        for i in indices {
            observed[i] += 1
        }
        provenance.push(identity);
        cursor = e;
    }
    if cursor != end {
        return Err("GLM files do not cover the full requested window".into());
    }
    let mut rows = Vec::new();
    for &(name, density) in PRODUCTS {
        let field = arm
            .fields
            .get(name)
            .ok_or_else(|| format!("forecast diagnostic {name} absent"))?;
        if field.len() != cells || field.par_iter().any(|x| x.is_finite() && *x < 0.) {
            return Err(format!("invalid {name} grid"));
        }
        let counts: Vec<f64> = field
            .par_iter()
            .zip(&areas)
            .zip(&eligible)
            .map(|((v, a), eligible)| {
                if !eligible {
                    f64::NAN
                } else if density {
                    v * a
                } else {
                    v * 300.
                }
            })
            .collect();
        rows.push(json!({"product":name,"forecast_conversion":if density{"five-minute flash density * native cell area"}else{"flash rate per cell * 300 s"},"scores":count_score(&counts,&observed)?}));
    }
    Ok(
        json!({"schema":RECEIPT_SCHEMA,"valid_time":request.valid_time,"window_start":request.window_start,"window_end":request.window_end,"verification_bbox":request.bbox,"model_source":arm.provenance,"area_source":verification_io::source_identity(&request.area_source_path)?,"area_method":"DX * DY / MAPFAC_M^2 / 1e6 km2","quality":"GLM flash_quality_flag == 0; flash centroids assigned to nearest native cell after containment","temporal_support":"instantaneous model threat held over 300 s; observed flashes grouped by contiguous 20 s product coverage; not a time-integrated model forecast","observation_source":"NOAA GOES GLM L2 LCFA; total lightning, not cloud-to-ground only","observation_terms":"NOAA NODD open public data; https://registry.opendata.aws/noaa-goes/; legacy file license attribute retained per file","glm_sources":provenance,"rows":rows,"rayon_threads":rayon::current_num_threads()}),
    )
}
#[cfg(test)]
mod tests {
    #[test]
    fn count_errors_preserve_missing_support() {
        let r = super::count_score(&[2., f64::NAN, 0.], &[1, 5, 2]).unwrap();
        assert_eq!(r["common_cells"], 2);
        assert_eq!(r["observed_flashes"], 3.);
        assert_eq!(r["forecast_flashes"], 2.);
        assert_eq!(r["bias_flashes_per_cell"], -0.5);
        assert_eq!(r["rmse_flashes_per_cell"], 2.5_f64.sqrt());
    }
}
