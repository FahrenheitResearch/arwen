//! Incident discovery, observed perimeter and satellite active-pixel admission.
//! Discovery is reported incident time, not measured physical ignition.
//! Active pixel footprints do not measure the complete burned perimeter.
use crate::error::{Result, StaticError};
use crate::projection::{GridSpec, ProjectedGrid};
use crate::sfire::{self, FirePerimeterRequest, FireStaticLoadRequest};
use crate::types::FieldSet;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
fn invalid(message: impl Into<String>) -> StaticError {
    StaticError::Invalid(message.into())
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BoundObservation {
    pub path: PathBuf,
    pub sha256: String,
    pub source_url: String,
    pub license: String,
}
impl BoundObservation {
    fn read(&self) -> Result<Vec<u8>> {
        if self.source_url.is_empty() || self.license.is_empty() {
            return Err(invalid(
                "Fire observations require their source URL and license",
            ));
        }
        let bytes = std::fs::read(&self.path)?;
        if hash(&bytes) != self.sha256.to_ascii_lowercase() {
            return Err(invalid(format!(
                "Fire source hash differs: {}",
                self.path.display()
            )));
        }
        Ok(bytes)
    }
}
/// Provider differences are field metadata, not per-provider adapters.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FeatureSource {
    #[serde(flatten)]
    pub input: BoundObservation,
    pub id_property: String,
    pub time_property: String,
}
#[derive(Debug, Deserialize)]
pub struct HotspotSource {
    #[serde(flatten)]
    pub input: BoundObservation,
    pub earliest_utc_epoch_ms: i64,
    /// Explicit incident search box [west, south, east, north] in degrees.
    pub bounds: [f64; 4],
    pub minimum_modis_confidence: f64,
    pub include_low_confidence_viirs: bool,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IgnitionRequest {
    pub grid_spec: GridSpec,
    pub sr_x: usize,
    pub sr_y: usize,
    pub incident: FeatureSource,
    pub incident_id: String,
    pub model_start_utc_epoch_ms: i64,
    pub initialization_utc_epoch_ms: i64,
    pub assumed_burn_age_seconds: f64,
    #[serde(default)]
    pub perimeters: Option<FeatureSource>,
    #[serde(default)]
    pub hotspots: Option<HotspotSource>,
    #[serde(default)]
    pub incident_radius_m: Option<f64>,
    pub output_directory: PathBuf,
    #[serde(default)]
    pub static_bundle: Option<PathBuf>,
    #[serde(default)]
    pub output: Option<PathBuf>,
}
fn normalize_id(id: &str) -> String {
    id.trim().trim_matches(['{', '}']).to_ascii_lowercase()
}
fn feature_collection(source: &FeatureSource) -> Result<Value> {
    let bytes = source.input.read()?;
    let mut value: Value = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
    // Acquisition pages remain opaque until this native decoder joins them.
    if value["type"] == "gpuwm-fire-geojson-pages-v1" {
        let pages = value["pages"]
            .as_array()
            .ok_or_else(|| invalid("Fire page manifest lacks pages"))?;
        if pages.is_empty() {
            return Err(invalid("Fire page manifest is empty"));
        }
        let mut features = Vec::new();
        for (i, page) in pages.iter().enumerate() {
            if page["type"] != "FeatureCollection" {
                return Err(invalid("Fire page must be GeoJSON FeatureCollection"));
            }
            if let Some(crs) = page.get("crs") {
                let text = crs.to_string();
                if !text.contains("4326") && !text.contains("CRS84") {
                    return Err(invalid("Fire page CRS is not EPSG:4326"));
                }
            }
            let truncated = page["exceededTransferLimit"].as_bool() == Some(true)
                || page["properties"]["exceededTransferLimit"].as_bool() == Some(true);
            if i + 1 == pages.len() && truncated {
                return Err(invalid("Final fire acquisition page remains truncated"));
            }
            features.extend(
                page["features"]
                    .as_array()
                    .ok_or_else(|| invalid("Fire page lacks features"))?
                    .clone(),
            );
        }
        value = json!({"type":"FeatureCollection","features":features});
    }
    if value.get("exceededTransferLimit").and_then(Value::as_bool) == Some(true)
        || value
            .get("properties")
            .and_then(|v| v.get("exceededTransferLimit"))
            .and_then(Value::as_bool)
            == Some(true)
    {
        return Err(invalid(
            "Fire response is truncated; acquire every ArcGIS page",
        ));
    }
    if value["type"] != "FeatureCollection" || !value["features"].is_array() {
        return Err(invalid(
            "Fire source must be EPSG:4326 GeoJSON FeatureCollection",
        ));
    }
    if let Some(crs) = value.get("crs") {
        let text = crs.to_string();
        if !text.contains("4326") && !text.contains("CRS84") {
            return Err(invalid("Fire GeoJSON CRS must be EPSG:4326 or OGC CRS84"));
        }
    }
    Ok(value)
}
fn selected<'a>(source: &FeatureSource, data: &'a Value, id: &str) -> Vec<&'a Value> {
    data["features"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| {
            f["properties"][&source.id_property]
                .as_str()
                .is_some_and(|v| normalize_id(v) == normalize_id(id))
        })
        .collect()
}
fn timestamp(feature: &Value, key: &str) -> Result<i64> {
    feature["properties"][key].as_i64().ok_or_else(||invalid(format!(
        "Fire observation has no measured UTC unix_ms timestamp in {key}; catalog modification time is not observation time")))
}
fn point(feature: &Value) -> Result<[f64; 2]> {
    let g = &feature["geometry"];
    if g["type"] != "Point" {
        return Err(invalid("Incident ignition requires Point geometry"));
    }
    let lon = g["coordinates"][0].as_f64().unwrap_or(f64::NAN);
    let lat = g["coordinates"][1].as_f64().unwrap_or(f64::NAN);
    if !lon.is_finite()
        || !lat.is_finite()
        || !(-180.0..=180.0).contains(&lon)
        || !(-90.0..=90.0).contains(&lat)
    {
        return Err(invalid("Incident point leaves finite EPSG:4326"));
    }
    Ok([lon, lat])
}
fn polygons(g: &Value) -> Result<Vec<Value>> {
    match g["type"].as_str() {
        Some("Polygon") => Ok(vec![g["coordinates"].clone()]),
        Some("MultiPolygon") => g["coordinates"]
            .as_array()
            .cloned()
            .ok_or_else(|| invalid("Invalid MultiPolygon")),
        _ => Err(invalid(
            "Observed fire geometry must be Polygon or MultiPolygon",
        )),
    }
}
fn circle(grid: &ProjectedGrid, lon: f64, lat: f64, radius: f64) -> Result<Value> {
    if !radius.is_finite() || radius <= 0. {
        return Err(invalid(
            "Incident ignition radius must be finite and positive",
        ));
    }
    let (x, y) = grid.latlon_to_ij(lat, lon);
    let ring: Vec<_> = (0..=64)
        .map(|i| {
            let a = (i % 64) as f64 * std::f64::consts::TAU / 64.;
            let (lat, lon) = grid.ij_to_latlon(
                x + radius * a.cos() / grid.spec.dx,
                y + radius * a.sin() / grid.spec.dy,
            );
            json!([lon, lat])
        })
        .collect();
    Ok(json!([ring]))
}
// RFC 4180 fields, escaped quotes and quoted newlines.
fn csv_records(text: &str) -> Result<Vec<Vec<String>>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = text.trim_start_matches('\u{feff}').chars().peekable();
    let mut quoted = false;
    while let Some(c) = chars.next() {
        if c == '"' {
            if quoted && chars.peek() == Some(&'"') {
                field.push('"');
                chars.next();
            } else if quoted || field.is_empty() {
                quoted = !quoted;
            } else {
                return Err(invalid("FIRMS CSV quote inside unquoted field"));
            }
        } else if !quoted && (c == ',' || c == '\n') {
            row.push(std::mem::take(&mut field));
            if c == '\n' {
                rows.push(std::mem::take(&mut row));
            }
        } else if c != '\r' || quoted {
            field.push(c);
        }
    }
    if quoted {
        return Err(invalid("FIRMS CSV unterminated quoted field"));
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}
fn firms_time(date: &str, time: &str) -> Result<i64> {
    use chrono::{NaiveDate, NaiveTime};
    let date = NaiveDate::parse_from_str(date, "%Y-%m-%d").map_err(|e| invalid(e.to_string()))?;
    let digits = time.trim();
    if digits.is_empty() || digits.len() > 4 || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return Err(invalid("FIRMS acq_time must be UTC HHMM"));
    }
    let hhmm: u32 = digits
        .parse()
        .map_err(|_| invalid("Invalid FIRMS UTC HHMM"))?;
    let time = NaiveTime::from_hms_opt(hhmm / 100, hhmm % 100, 0)
        .ok_or_else(|| invalid("Invalid FIRMS UTC HHMM"))?;
    Ok(date.and_time(time).and_utc().timestamp_millis())
}
fn hotspot_polygons(
    source: &HotspotSource,
    init: i64,
    grid: &ProjectedGrid,
) -> Result<(Vec<Value>, Value)> {
    if source.earliest_utc_epoch_ms > init
        || !source.minimum_modis_confidence.is_finite()
        || !(0.0..=100.0).contains(&source.minimum_modis_confidence)
    {
        return Err(invalid("Invalid FIRMS time window or confidence"));
    }
    let [west, south, east, north] = source.bounds;
    if source.bounds.iter().any(|v| !v.is_finite())
        || west >= east
        || south >= north
        || west < -180.
        || east > 180.
        || south < -90.
        || north > 90.
    {
        return Err(invalid(
            "FIRMS needs finite incident bounds [west,south,east,north]",
        ));
    }
    let bytes = source.input.read()?;
    let rows = csv_records(std::str::from_utf8(&bytes).map_err(|e| invalid(e.to_string()))?)?;
    let header = rows.first().ok_or_else(|| invalid("Empty FIRMS CSV"))?;
    let col = |key: &str| {
        header
            .iter()
            .position(|v| v == key)
            .ok_or_else(|| invalid(format!("FIRMS CSV lacks {key}")))
    };
    let (lon, lat, date, time, instrument, confidence, scan, track) = (
        col("longitude")?,
        col("latitude")?,
        col("acq_date")?,
        col("acq_time")?,
        col("instrument")?,
        col("confidence")?,
        col("scan")?,
        col("track")?,
    );
    let kind = header.iter().position(|v| v == "type");
    let mut result = Vec::new();
    let mut rejected = 0;
    let mut times = Vec::new();
    for row in rows
        .iter()
        .skip(1)
        .filter(|r| !r.iter().all(|v| v.trim().is_empty()))
    {
        if row.len() != header.len() {
            return Err(invalid("FIRMS row length differs from header"));
        }
        let stamp = firms_time(&row[date], &row[time])?;
        if stamp < source.earliest_utc_epoch_ms || stamp > init {
            rejected += 1;
            continue;
        }
        let accept = match row[instrument].to_ascii_uppercase().as_str() {
            "VIIRS" => match row[confidence].as_str() {
                "h" | "n" => true,
                "l" => source.include_low_confidence_viirs,
                _ => return Err(invalid("VIIRS confidence must be l, n or h")),
            },
            "MODIS" => {
                let c = row[confidence]
                    .parse::<f64>()
                    .map_err(|_| invalid("MODIS confidence is not numeric"))?;
                if !c.is_finite() || !(0.0..=100.0).contains(&c) {
                    return Err(invalid("MODIS confidence leaves 0..100"));
                }
                c >= source.minimum_modis_confidence
            }
            _ => return Err(invalid("FIRMS instrument must be VIIRS or MODIS")),
        };
        if !accept || kind.is_some_and(|k| row[k] != "0") {
            rejected += 1;
            continue;
        }
        let number = |i: usize| {
            row[i]
                .parse::<f64>()
                .map_err(|_| invalid(format!("Invalid FIRMS number in {}", header[i])))
        };
        let (lon, lat, scan, track) = (number(lon)?, number(lat)?, number(scan)?, number(track)?);
        if !lon.is_finite()
            || !lat.is_finite()
            || !(-180.0..=180.0).contains(&lon)
            || !(-90.0..=90.0).contains(&lat)
            || !scan.is_finite()
            || !track.is_finite()
            || scan <= 0.
            || track <= 0.
        {
            return Err(invalid("FIRMS coordinates or footprint invalid"));
        }
        if lon < west || lon > east || lat < south || lat > north {
            rejected += 1;
            continue;
        }
        // CSV supplies size in km but no footprint orientation.
        result.push(circle(
            grid,
            lon,
            lat,
            (scan * track * 1e6 / std::f64::consts::PI).sqrt(),
        )?);
        times.push(stamp);
    }
    let metadata = json!({"accepted":result.len(),"rejected":rejected,"acquisition_utc_epoch_ms":times,
        "geometry_policy":"equal_area_circles_from_scan_track_km_orientation_unavailable",
        "earliest_utc_epoch_ms":source.earliest_utc_epoch_ms,"latest_utc_epoch_ms":init,
        "minimum_modis_confidence":source.minimum_modis_confidence,"include_low_confidence_viirs":source.include_low_confidence_viirs,
        "meaning":"active_fire_pixel_footprints_not_complete_burned_perimeter","bounds":source.bounds,"input":source.input});
    Ok((result, metadata))
}

pub fn prepare(r: &IgnitionRequest) -> Result<(FieldSet, Value)> {
    if r.initialization_utc_epoch_ms < r.model_start_utc_epoch_ms
        || !r.assumed_burn_age_seconds.is_finite()
        || r.assumed_burn_age_seconds < 0.
    {
        return Err(invalid(
            "Fire initialization requires nonnegative model time and explicit finite burn age",
        ));
    }
    let grid = ProjectedGrid::new(r.grid_spec.clone())?;
    let incidents = feature_collection(&r.incident)?;
    let matches = selected(&r.incident, &incidents, &r.incident_id);
    if matches.len() != 1 {
        return Err(invalid(
            "Incident ID must select exactly one public incident",
        ));
    }
    let incident = matches[0];
    let location = point(incident)?;
    let discovery = timestamp(incident, &r.incident.time_property)?;
    if discovery > r.initialization_utc_epoch_ms {
        return Err(invalid("Incident discovery is later than initialization"));
    }
    let mut geometry = Vec::new();
    let mut snapshots = Vec::new();
    let mut perim_time = None;
    if let Some(source) = &r.perimeters {
        let data = feature_collection(source)?;
        for f in selected(source, &data, &r.incident_id) {
            snapshots.push((
                timestamp(f, &source.time_property)?,
                polygons(&f["geometry"])?,
            ));
        }
        perim_time = snapshots
            .iter()
            .map(|(t, _)| *t)
            .filter(|t| *t <= r.initialization_utc_epoch_ms)
            .max();
        let stamp = perim_time.ok_or_else(|| {
            invalid("No measured incident perimeter exists at or before initialization")
        })?;
        for (_, p) in snapshots.iter().filter(|(t, _)| *t == stamp) {
            geometry.extend(p.clone());
        }
    }
    let hotspot_metadata = if let Some(source) = &r.hotspots {
        let (p, m) = hotspot_polygons(source, r.initialization_utc_epoch_ms, &grid)?;
        geometry.extend(p);
        Some(m)
    } else {
        None
    };
    let point_only = r.perimeters.is_none() && r.hotspots.is_none();
    if point_only {
        geometry.push(circle(
            &grid,
            location[0],
            location[1],
            r.incident_radius_m
                .ok_or_else(|| invalid("Point ignition needs explicit incident_radius_m"))?,
        )?);
    }
    if geometry.is_empty() {
        return Err(invalid(
            "No accepted perimeter or active-fire footprint initializes this grid",
        ));
    }
    let elapsed = (r.initialization_utc_epoch_ms - r.model_start_utc_epoch_ms) as f64 / 1000.;
    let age = r.assumed_burn_age_seconds
        + (r.initialization_utc_epoch_ms - perim_time.unwrap_or(r.initialization_utc_epoch_ms))
            as f64
            / 1000.;
    let observed = json!({"type":"FeatureCollection","features":[{
        "type":"Feature","properties":{"incident_id":r.incident_id,"initialization_utc_epoch_ms":r.initialization_utc_epoch_ms},
        "geometry":{"type":"MultiPolygon","coordinates":geometry}}]});
    sfire::perimeter_rings(&observed["features"][0]["geometry"], &grid)?;
    std::fs::create_dir_all(&r.output_directory)?;
    let geojson = r.output_directory.join("initialization.geojson");
    let contract_path = r.output_directory.join("initialization-contract.json");
    let bytes = serde_json::to_vec_pretty(&observed).map_err(|e| invalid(e.to_string()))?;
    let contract = json!({"input_sha256":hash(&bytes),"coordinate_reference_system":"EPSG:4326",
        "time_format":"unix_ms","time_property":"initialization_utc_epoch_ms",
        "geometry_policy":if point_only {"incident_point_with_explicit_radius"} else {"union_of_observed_polygons_and_declared_active_pixel_footprints"},
        "incident":r.incident,"perimeters":r.perimeters,"hotspots":hotspot_metadata,
        "perimeter_observation_utc_epoch_ms":perim_time,"initialization_utc_epoch_ms":r.initialization_utc_epoch_ms});
    std::fs::write(&geojson, bytes)?;
    std::fs::write(
        &contract_path,
        serde_json::to_vec_pretty(&contract).map_err(|e| invalid(e.to_string()))?,
    )?;
    let perimeter = FirePerimeterRequest {
        grid_spec: r.grid_spec.clone(),
        sr_x: r.sr_x,
        sr_y: r.sr_y,
        geojson,
        source_contract: contract_path,
        observed_utc_epoch_ms: r.initialization_utc_epoch_ms,
        model_elapsed_seconds: elapsed,
        assumed_burn_age_seconds: age,
        output: None,
    };
    let mut fields = sfire::observed_perimeter(&perimeter)?;
    let mut metadata: Value = serde_json::from_str(&fields.coverage_reports["SFIRE_OBSERVED"])
        .map_err(|e| invalid(e.to_string()))?;
    // The composed geometry is a declared model initialization, not a new
    // measurement. Preserve the provider's measurement time independently.
    metadata["initialization_utc_epoch_ms"] = metadata["observed_utc_epoch_ms"].take();
    metadata["observed_utc_epoch_ms"] = json!(perim_time.unwrap_or(discovery));
    metadata["initialization_geometry_meaning"] =
        json!("declared_initial_state_not_a_new_observation");
    metadata["source_contract"] = contract;
    metadata["admission"] = json!("public_fire_observations_v1");
    metadata["incident_id"] = json!(r.incident_id);
    metadata["incident_point_lon_lat"] = json!(location);
    metadata["incident_discovery_utc_epoch_ms"] = json!(discovery);
    metadata["ignition_time_meaning"] = json!("reported_incident_discovery_not_physical_ignition");
    metadata["model_start_utc_epoch_ms"] = json!(r.model_start_utc_epoch_ms);
    metadata["parallelism"] = json!({"signed_distance_threads":rayon::current_num_threads()});
    metadata["observed_perimeter_request"] =
        serde_json::to_value(&perimeter).map_err(|e| invalid(e.to_string()))?;
    if let Some(path) = &r.static_bundle {
        let (mut statics, mut bundle) = sfire::load(&FireStaticLoadRequest {
            path: path.clone(),
            expected_sha256: None,
            expected_grid_spec: Some(r.grid_spec.clone()),
        })?;
        if bundle.sr_x != r.sr_x || bundle.sr_y != r.sr_y {
            return Err(invalid("Ignition refinement differs from static bundle"));
        }
        statics.fields.extend(fields.fields);
        fields = statics;
        bundle.observed_initialization = Some(metadata.clone());
        bundle.field_sha256 = fields
            .fields
            .iter()
            .map(|(k, f)| (k.clone(), sfire::field_digest(f)))
            .collect();
        let output = r
            .output
            .as_ref()
            .ok_or_else(|| invalid("Attaching ignition to static bundle requires output"))?;
        if output == path
            || (output.exists() && std::fs::canonicalize(output)? == std::fs::canonicalize(path)?)
        {
            return Err(invalid(
                "Ignition output must differ from input static bundle",
            ));
        }
        crate::npz::write_deterministic_npz_extra(
            output,
            &fields,
            &[(
                "SFIRE.json".into(),
                serde_json::to_vec(&bundle).map_err(|e| invalid(e.to_string()))?,
            )]
            .into(),
        )?;
        metadata["output"] = json!(output);
        metadata["output_sha256"] = json!(hash(&std::fs::read(output)?));
    } else if r.output.is_some() {
        return Err(invalid(
            "Ignition output requires admitted terrain/fuel static_bundle",
        ));
    }
    std::fs::write(
        r.output_directory.join("ignition-receipt.json"),
        serde_json::to_vec_pretty(&metadata).map_err(|e| invalid(e.to_string()))?,
    )?;
    Ok((fields, metadata))
}

/// Compare only equal valid times. Both planes use the actual output coordinates.
pub fn verification(
    source: &FeatureSource,
    id: &str,
    observed_ms: i64,
    model_ms: i64,
    spec: &GridSpec,
    lat: &[f32],
    lon: &[f32],
    lfn: &[f32],
    cell_area: f64,
) -> Result<(Vec<f32>, Value)> {
    if observed_ms != model_ms {
        return Err(invalid(
            "Perimeter verification requires observation and forecast at the same UTC time",
        ));
    }
    if lat.len() != lon.len()
        || lat.len() != lfn.len()
        || lat.is_empty()
        || !cell_area.is_finite()
        || cell_area <= 0.
    {
        return Err(invalid("Perimeter verification has inconsistent geometry"));
    }
    let data = feature_collection(source)?;
    let grid = ProjectedGrid::new(spec.clone())?;
    let mut polys = Vec::new();
    for f in selected(source, &data, id) {
        if timestamp(f, &source.time_property)? == observed_ms {
            polys.extend(sfire::perimeter_rings(&f["geometry"], &grid)?);
        }
    }
    if polys.is_empty() {
        return Err(invalid(
            "No independently measured incident perimeter matches forecast valid time",
        ));
    }
    let observed: Result<Vec<f32>> = lat
        .par_iter()
        .zip(lon)
        .zip(lfn)
        .map(|((&lat, &lon), &model)| {
            if !lat.is_finite() || !lon.is_finite() || !model.is_finite() {
                return Err(invalid("Perimeter verification contains nonfinite values"));
            }
            let (x, y) = grid.latlon_to_ij(lat as f64, lon as f64);
            Ok(
                sfire::perimeter_distance([(x - 0.5) * spec.dx, (y - 0.5) * spec.dy], &polys)
                    as f32,
            )
        })
        .collect();
    let observed = observed?;
    let counts = lfn
        .par_iter()
        .zip(&observed)
        .map(|(&m, &o)| {
            [
                usize::from(m <= 0. && o <= 0.),
                usize::from(m <= 0. && o > 0.),
                usize::from(m > 0. && o <= 0.),
                usize::from(m <= 0.),
                usize::from(o <= 0.),
            ]
        })
        .reduce(|| [0; 5], |a, b| std::array::from_fn(|i| a[i] + b[i]));
    let union = counts[0] + counts[1] + counts[2];
    let receipt = json!({"schema":"gpuwm-fire-perimeter-verification-v1","incident_id":id,
        "observation_utc_epoch_ms":observed_ms,"model_valid_utc_epoch_ms":model_ms,"source":source,
        "intersection_cells":counts[0],"false_alarm_cells":counts[1],"missed_cells":counts[2],
        "intersection_over_union":if union>0 {Some(counts[0] as f64/union as f64)} else {None},
        "model_area_m2":counts[3] as f64*cell_area,"observed_area_m2":counts[4] as f64*cell_area,
        "area_meaning":"nominal_projected_grid_cell_area_without_map_factor_correction",
        "scope":"cell_center_geometry_on_rendered_physical_grid","threads":rayon::current_num_threads()});
    Ok((observed, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn firms_utc_and_quoted_records() {
        assert_eq!(firms_time("2021-12-30", "1838").unwrap(), 1640889480000);
        assert!(firms_time("2021-02-29", "1200").is_err());
        assert!(firms_time("2021-12-30", "2400").is_err());
        assert_eq!(
            csv_records("a,b\r\n\"x,y\",\"z\"\"q\"\r\n").unwrap()[1],
            vec!["x,y", "z\"q"]
        );
        assert!(csv_records("a\n\"unclosed").is_err());
    }
    #[test]
    fn catalog_time_is_not_measurement_time() {
        let f = json!({"properties":{"poly_PolygonDateTime":null,"poly_DateCurrent":1666364144584_i64}});
        assert!(
            timestamp(&f, "poly_PolygonDateTime")
                .unwrap_err()
                .to_string()
                .contains("catalog")
        );
        assert_eq!(normalize_id("{ABC-D}"), "abc-d");
    }
    fn grid_spec() -> GridSpec {
        GridSpec {
            kind: crate::projection::ProjectionKind::Mercator,
            ref_lat: 0.,
            ref_lon: 0.,
            truelat1: 0.,
            truelat2: 0.,
            stand_lon: 0.,
            dx: 100.,
            dy: 100.,
            e_we: 9,
            e_sn: 9,
            known_x: 4.5,
            known_y: 4.5,
            moad_cen_lat: 0.,
            moad_cen_lon: 0.,
            lat_deg: vec![],
            lon0_deg: 0.,
            dlon_deg: 0.,
        }
    }
    fn fixture() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("fire-observations-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
    fn bound(dir: &std::path::Path, name: &str, bytes: &[u8]) -> BoundObservation {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        BoundObservation {
            path,
            sha256: hash(bytes),
            source_url: "https://example.invalid/captured-test-fixture".into(),
            license: "Public Domain".into(),
        }
    }
    fn square(grid: &ProjectedGrid, lo: f64, hi: f64) -> Value {
        Value::Array(
            [(lo, lo), (hi, lo), (hi, hi), (lo, hi), (lo, lo)]
                .iter()
                .map(|&(x, y)| {
                    let (lat, lon) = grid.ij_to_latlon(x, y);
                    json!([lon, lat])
                })
                .collect(),
        )
    }
    #[test]
    fn real_source_admission_keeps_holes_and_holds_out_future_perimeters() {
        let dir = fixture();
        let spec = grid_spec();
        let grid = ProjectedGrid::new(spec.clone()).unwrap();
        let incident = json!({"type":"FeatureCollection","features":[{"type":"Feature",
            "properties":{"id":"{ABC}","discovery":1000},"geometry":{"type":"Point","coordinates":[0.,0.]}}]});
        let data = json!({"type":"FeatureCollection","features":[
            {"type":"Feature","properties":{"id":"abc","measured":2000},
             "geometry":{"type":"Polygon","coordinates":[square(&grid,2.,6.),square(&grid,3.,4.)]}},
            {"type":"Feature","properties":{"id":"abc","measured":4000},
             "geometry":{"type":"Polygon","coordinates":[square(&grid,1.,8.)]}}]});
        let incident = FeatureSource {
            input: bound(
                &dir,
                "incident.json",
                &serde_json::to_vec(&incident).unwrap(),
            ),
            id_property: "id".into(),
            time_property: "discovery".into(),
        };
        let perimeters = FeatureSource {
            input: bound(&dir, "perimeters.json", &serde_json::to_vec(&data).unwrap()),
            id_property: "id".into(),
            time_property: "measured".into(),
        };
        let request = IgnitionRequest {
            grid_spec: spec.clone(),
            sr_x: 2,
            sr_y: 2,
            incident,
            incident_id: "ABC".into(),
            model_start_utc_epoch_ms: 1000,
            initialization_utc_epoch_ms: 3000,
            assumed_burn_age_seconds: 30.,
            perimeters: Some(perimeters.clone()),
            hotspots: None,
            incident_radius_m: None,
            output_directory: dir.join("initial"),
            static_bundle: None,
            output: None,
        };
        let (fields, receipt) = prepare(&request).unwrap();
        assert_eq!(receipt["initialization_utc_epoch_ms"], 3000);
        assert_eq!(receipt["observed_utc_epoch_ms"], 2000);
        assert_eq!(receipt["incident_discovery_utc_epoch_ms"], 1000);
        let lfn = fields.fields["LFN_HIST"].data();
        let at = |x: f64, y: f64| {
            lfn[(((y - 0.5) * 2. - 0.5).round() as usize) * 16
                + ((x - 0.5) * 2. - 0.5).round() as usize]
        };
        assert!(at(2.25, 2.25) < 0., "outer burned ring");
        assert!(at(3.25, 3.25) > 0., "unburned island retained");
        assert!(
            at(7.25, 7.25) > 0.,
            "future perimeter must not leak into initialization"
        );
        let (lat, lon) = sfire::coordinates(&spec, 2, 2, 0).unwrap();
        let lat: Vec<f32> = lat.data.iter().map(|v| *v as f32).collect();
        let lon: Vec<f32> = lon.data.iter().map(|v| *v as f32).collect();
        let modeled: Vec<f32> = lfn.iter().map(|v| *v as f32).collect();
        assert!(
            verification(
                &perimeters,
                "abc",
                4000,
                3000,
                &spec,
                &lat,
                &lon,
                &modeled,
                2500.
            )
            .is_err()
        );
        let (_, score) = verification(
            &perimeters,
            "abc",
            4000,
            4000,
            &spec,
            &lat,
            &lon,
            &modeled,
            2500.,
        )
        .unwrap();
        assert!(score["missed_cells"].as_u64().unwrap() > 0);
        assert!(score["intersection_over_union"].as_f64().unwrap() < 1.);
        std::fs::write(&request.incident.input.path, b"changed").unwrap();
        assert!(
            prepare(&request)
                .unwrap_err()
                .to_string()
                .contains("hash differs")
        );
    }
    #[test]
    fn overlapping_pixels_do_not_create_internal_zero_contours() {
        let a = vec![vec![[0., 0.], [4., 0.], [4., 4.], [0., 4.], [0., 0.]]];
        let b = vec![vec![[2., 0.], [6., 0.], [6., 4.], [2., 4.], [2., 0.]]];
        assert_eq!(sfire::perimeter_distance([4., 2.], &[a, b]), -2.);
    }
    #[test]
    fn satellite_pixels_require_time_confidence_and_incident_bounds() {
        let dir = fixture();
        let grid = ProjectedGrid::new(grid_spec()).unwrap();
        let bytes=b"latitude,longitude,acq_date,acq_time,instrument,confidence,scan,track,type\n0,0,2021-12-30,1838,VIIRS,n,0.375,0.375,0\n0,0,2021-12-30,1839,VIIRS,h,0.375,0.375,0\n0,1,2021-12-30,1838,MODIS,100,1,1,0\n0,0,2021-12-30,1838,VIIRS,l,0.375,0.375,0\n0,0,2021-12-30,1838,MODIS,25,1,1,0\n";
        let request = HotspotSource {
            input: bound(&dir, "firms.csv", bytes),
            earliest_utc_epoch_ms: 1640889480000,
            bounds: [-0.1, -0.1, 0.1, 0.1],
            minimum_modis_confidence: 50.,
            include_low_confidence_viirs: false,
        };
        let (polygons, receipt) = hotspot_polygons(&request, 1640889480000, &grid).unwrap();
        assert_eq!(polygons.len(), 1);
        assert_eq!(receipt["rejected"], 4);
        let footprint = sfire::perimeter_rings(
            &json!({"type":"MultiPolygon","coordinates":polygons}),
            &grid,
        )
        .unwrap();
        let radius = (375.0_f64 * 375. / std::f64::consts::PI).sqrt();
        let (x, y) = grid.latlon_to_ij(0., 0.);
        assert!(
            (sfire::perimeter_distance([(x - 0.5) * 100., (y - 0.5) * 100.], &footprint).abs()
                - radius)
                .abs()
                < 1.
        );
    }
}
