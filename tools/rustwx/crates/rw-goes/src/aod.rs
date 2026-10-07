//! ABI AOD at 550 nm, on the source fixed grid with ungated DQF.
//! DQF meanings are read from the source NetCDF and recorded in every pack.
//! Default gate keeps only 0 (high), rejecting 1 (medium), 2 (low), 3 (none).
use super::*;
use crate::pack::{ContainerMeta, decode_container, write_container};
use serde::{Deserialize, Serialize};
pub const AOD_SCHEMA: &str = "gpuwm-obs.goes-aod.v1";
#[derive(Debug, Serialize, Deserialize)]
pub struct AodMeta {
    pub schema: String,
    pub status: String,
    pub satellite: String,
    pub sector: String,
    pub scan_start: String,
    pub scan_end: String,
    pub sources: Vec<SourceEntry>,
    pub window: Option<[usize; 4]>,
    pub projection: ProjectionEntry,
    pub nx: usize,
    pub ny: usize,
    pub x_scan_rad: Vec<f64>,
    pub y_scan_rad: Vec<f64>,
    pub planes: BTreeMap<String, String>,
    pub plane_order: Vec<String>,
    pub arrays: BTreeMap<String, ArrayEntry>,
    pub payload_bytes: usize,
    pub content_sha256: String,
    pub units: String,
    pub wavelength_nm: u32,
    pub dqf_flag_meanings: String,
    pub dqf_flag_values: Vec<f64>,
}
impl ContainerMeta for AodMeta {
    const WRITTEN_SCHEMA: &'static str = AOD_SCHEMA;
    const READABLE_SCHEMAS: &'static [&'static str] = &[AOD_SCHEMA];
    fn schema(&self) -> &str {
        &self.schema
    }
    fn content_sha256(&self) -> &str {
        &self.content_sha256
    }
    fn arrays(&self) -> &BTreeMap<String, ArrayEntry> {
        &self.arrays
    }
}
pub fn cmd_aod(o: &Options) -> Result<String, Box<dyn Error>> {
    let path = o
        .aod
        .as_deref()
        .ok_or_else(|| boxed_error("--aod FILE required"))?;
    let out = o
        .out
        .as_deref()
        .ok_or_else(|| boxed_error("--out FILE required"))?;
    let source = decode_source(path, CloudProduct::AerosolOpticalDepth, o.window)?;
    let scene = source.scene();
    if !["ABI-L2-AODC", "ABI-L2-AODF"].contains(&scene.product.as_str()) {
        return Err(boxed_error("not an AOD C/F granule"));
    }
    if source.decoded.field.units.as_deref() != Some("1") {
        return Err(boxed_error("AOD units must be 1"));
    }
    let file = rw_sat::netcdf::open_goes_netcdf_lossy(path)?;
    let meanings = file
        .variable("DQF")
        .and_then(|v| {
            v.attribute("flag_meanings")
                .and_then(|a| a.as_string())
                .map(str::to_string)
        })
        .ok_or_else(|| boxed_error("AOD DQF flag_meanings missing"))?;
    let flags = file
        .variable("DQF")
        .and_then(|v| {
            v.attribute("flag_values")
                .and_then(|a| a.value().as_f64_vec())
        })
        .ok_or_else(|| boxed_error("AOD DQF flag_values missing"))?;
    if flags != vec![0.0, 1.0, 2.0, 3.0] {
        return Err(boxed_error(format!(
            "unrecognized AOD DQF values: {flags:?}"
        )));
    }
    let normalized = meanings.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized
        != "high_quality_retrieval_qf medium_quality_retrieval_qf low_quality_retrieval_qf no_retrieval_qf"
    {
        return Err(boxed_error(format!(
            "unrecognized AOD DQF mapping: {normalized}"
        )));
    }
    let shape = [scene.fixed_grid.ny, scene.fixed_grid.nx];
    let mut builder = PayloadBuilder::new();
    let mut planes = BTreeMap::new();
    let mut order = vec![];
    let (lat, lon) = scene.lat_lon_mesh();
    for (name, values) in [
        ("aod", source.decoded.field.values.as_slice()),
        ("aod_dqf", source.dqf_values.as_slice()),
        ("lat", lat.as_slice()),
        ("lon", lon.as_slice()),
    ] {
        push_plane(&mut builder, &mut planes, &mut order, name, values, &shape)?;
    }
    let (payload, arrays) = builder.finish();
    let meta = AodMeta {
        schema: AOD_SCHEMA.into(),
        status: "READY".into(),
        satellite: scene.satellite.as_str().into(),
        sector: sector_token_from_scene(&scene.sector),
        scan_start: iso8601(scene.start_time_utc),
        scan_end: iso8601(scene.end_time_utc),
        sources: vec![source_entry(&source)],
        window: o.window,
        projection: ProjectionEntry {
            perspective_point_height_m: scene.projection.perspective_point_height_m,
            semi_major_axis_m: scene.projection.semi_major_axis_m,
            semi_minor_axis_m: scene.projection.semi_minor_axis_m,
            longitude_of_projection_origin_deg: scene.projection.longitude_of_projection_origin_deg,
            sweep_angle_axis: scene.projection.sweep_angle_axis.as_str().into(),
        },
        nx: shape[1],
        ny: shape[0],
        x_scan_rad: scene.fixed_grid.x_scan_rad.clone(),
        y_scan_rad: scene.fixed_grid.y_scan_rad.clone(),
        planes,
        plane_order: order,
        arrays,
        payload_bytes: payload.len(),
        content_sha256: hex_sha256(&payload),
        units: "1".into(),
        wavelength_nm: 550,
        dqf_flag_meanings: normalized,
        dqf_flag_values: flags.clone(),
    };
    let bytes = write_container(out, &meta, &payload)?;
    Ok(format!(
        "{}\n",
        serde_json::to_string_pretty(
            &serde_json::json!({"schema":"gpuwm-obs.goes-aod-build.v1","status":"READY","pack":out,"bytes":bytes,"content_sha256":meta.content_sha256,"dqf":meta.sources[0].dqf,"dqf_flag_meanings":meta.dqf_flag_meanings,"dqf_flag_values":flags})
        )?
    ))
}
pub fn verify(bytes: &[u8]) -> Result<String, Box<dyn Error>> {
    let (meta, payload): (AodMeta, Vec<u8>) = decode_container(bytes)?;
    check_grid_and_planes(
        meta.nx,
        meta.ny,
        meta.x_scan_rad.len(),
        meta.y_scan_rad.len(),
        &meta.plane_order,
        &meta.planes,
        &meta.arrays,
    )?;
    for name in ["aod", "aod_dqf", "lat", "lon"] {
        if !meta.planes.contains_key(name) {
            return Err(boxed_error(format!("missing AOD plane {name}")));
        }
    }
    if meta.status != "READY"
        || meta.units != "1"
        || meta.wavelength_nm != 550
        || meta.dqf_flag_values != vec![0.0, 1.0, 2.0, 3.0]
    {
        return Err(boxed_error(
            "invalid AOD status, units, wavelength or DQF mapping",
        ));
    }
    let a = &meta.arrays[&meta.planes["aod"]];
    let q = &meta.arrays[&meta.planes["aod_dqf"]];
    for i in 0..meta.nx * meta.ny {
        let value = f32::from_le_bytes(payload[a.offset + i * 4..a.offset + i * 4 + 4].try_into()?);
        let dqf = f32::from_le_bytes(payload[q.offset + i * 4..q.offset + i * 4 + 4].try_into()?);
        if value.is_finite() && dqf != 0.0 {
            return Err(boxed_error("AOD retains non-high-quality pixel"));
        }
    }
    if meta.payload_bytes != payload.len() {
        return Err(boxed_error("AOD payload size mismatch"));
    }
    Ok(format!(
        "{}\n",
        serde_json::json!({"schema":"gpuwm-obs.goes-aod-verify.v1","status":"VERIFIED","content_sha256":meta.content_sha256})
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_crop_roundtrip_and_corruption() {
        let hex = include_str!("../tests/data/aod-crop.hex").trim();
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        assert!(verify(&bytes).unwrap().contains("VERIFIED"));
        let mut corrupt = bytes.to_vec();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        assert!(verify(&corrupt).is_err());
    }
    #[test]
    fn aod_products_and_gate() {
        let p = CloudProduct::AerosolOpticalDepth;
        assert_eq!(p.abi_product(Sector::Conus).unwrap(), "ABI-L2-AODC");
        assert_eq!(p.abi_product(Sector::FullDisk).unwrap(), "ABI-L2-AODF");
        assert!(p.abi_product(Sector::Meso1).is_none());
        assert!(p.dqf_rule().is_good(0.0));
        for dqf in [1.0, 2.0, 3.0, f32::NAN] {
            assert!(!p.dqf_rule().is_good(dqf));
        }
    }
}
