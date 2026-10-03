/// Table-selected model state in model-independent physical units.
/// Horizontal fields use row-major Y/X order; volume fields use Z/Y/X order.
pub struct ModelRadarFields {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub latitude_deg: Vec<f32>,
    pub longitude_deg: Vec<f32>,
    pub height_msl_m: Vec<f32>,
    pub reflectivity_dbz: Vec<f32>,
    pub eastward_wind_mps: Vec<f32>,
    pub northward_wind_mps: Vec<f32>,
    pub upward_wind_mps: Vec<f32>,
    pub terrain_msl_m: Vec<f32>,
    pub grid_spacing_m: Option<f64>,
    pub model_label: String,
}

impl WrfRadarFields {
    /// Accept normalized physical arrays from any model, without choosing an
    /// ingest path by model name. Missing samples stay NaN in the forward model.
    pub fn from_model_fields(fields: ModelRadarFields) -> Result<Self, String> {
        if fields.nx < 2 || fields.ny < 2 || fields.nz < 2 {
            return Err("radar sampling needs at least two cells on each grid axis".to_owned());
        }
        let horizontal = fields.nx.checked_mul(fields.ny)
            .ok_or_else(|| "radar horizontal grid size overflow".to_owned())?;
        let cells = horizontal.checked_mul(fields.nz)
            .ok_or_else(|| "radar volume grid size overflow".to_owned())?;
        for (name, actual, expected) in [
            ("latitude", fields.latitude_deg.len(), horizontal),
            ("longitude", fields.longitude_deg.len(), horizontal),
            ("terrain", fields.terrain_msl_m.len(), horizontal),
            ("height", fields.height_msl_m.len(), cells),
            ("reflectivity", fields.reflectivity_dbz.len(), cells),
            ("eastward wind", fields.eastward_wind_mps.len(), cells),
            ("northward wind", fields.northward_wind_mps.len(), cells),
            ("upward wind", fields.upward_wind_mps.len(), cells),
        ] {
            if actual != expected {
                return Err(format!("radar {name} has {actual} samples; expected {expected}"));
            }
        }
        let lut = InverseLut::build_with_shape_domain_bounded(
            &fields.latitude_deg, &fields.longitude_deg, fields.nx, fields.ny,
        ).ok_or_else(|| "radar geolocation contains no usable domain".to_owned())?;
        Ok(Self {
            nx: fields.nx, ny: fields.ny, nz: fields.nz,
            lat: fields.latitude_deg, lon: fields.longitude_deg,
            height_msl: fields.height_msl_m, dbz: fields.reflectivity_dbz,
            u: fields.eastward_wind_mps, v: fields.northward_wind_mps,
            w: fields.upward_wind_mps, terrain_m: fields.terrain_msl_m,
            property_scattering: None, property_table_identity: None,
            property_table_resident_bytes: 0, raw_property_scene: None,
            refractivity_model: None, polarimetric: None,
            dual_pol_status: Some("Scalar model fields provide reflectivity and radial velocity; native microphysics is required for dual-pol".to_owned()),
            tke_tenths_m2s2: None, ref_source: "model-native reflectivity",
            dx_m: fields.grid_spacing_m, source_model_override: Some(fields.model_label),
            source_microphysics_override: None, source_scattering_override: None,
            source_provenance_fragment: Some("metadata-selected physical fields; earth-relative winds; height MSL".to_owned()),
            lut,
        })
    }
}

/// Build with validation and return a failure instead of panicking.
pub fn try_build_synthetic_volume(
    fields: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
) -> Result<RadarVolume, String> {
    try_build_synthetic_volume_reporting(fields, valid_time, config, &|_| {})
}
/// Read scalar and supported microphysical fields with the same operator used
/// by the interactive app, including its capability and provenance checks.
pub fn read_wrf_radar_fields_for_config(
    file: &WrfFile,
    source_identity: &WrfSourceIdentity,
    timeidx: usize,
    config: &SyntheticRadarConfig,
) -> Result<WrfRadarFields, String> {
    read_wrf_radar_fields_for_config_reporting(
        file, source_identity, timeidx, config, &|_| {}, 0, None,
    )
}
#[cfg(test)]
mod model_input_tests {
    use super::*;

    fn uniform_model() -> ModelRadarFields {
        ModelRadarFields {
            nx: 2, ny: 2, nz: 2,
            latitude_deg: vec![34.0, 34.0, 36.0, 36.0],
            longitude_deg: vec![-99.0, -97.0, -99.0, -97.0],
            height_msl_m: [vec![0.0; 4], vec![10_000.0; 4]].concat(),
            reflectivity_dbz: vec![40.0; 8],
            eastward_wind_mps: vec![10.0; 8],
            northward_wind_mps: vec![0.0; 8],
            upward_wind_mps: vec![0.0; 8],
            terrain_msl_m: vec![0.0; 4], grid_spacing_m: Some(3000.0),
            model_label: "WOOF".to_owned(),
        }
    }

    #[test]
    fn generic_fields_preserve_reflectivity_and_velocity_sign_in_real_volume() {
        let fields = WrfRadarFields::from_model_fields(uniform_model()).unwrap();
        let mut config = SyntheticRadarConfig::default();
        config.site_id = "TEST".to_owned();
        config.site_lat_deg = Some(35.0);
        config.site_lon_deg = Some(-98.0);
        config.antenna_msl_m = Some(100.0);
        config.elevations_deg = vec![0.5];
        config.azimuth_count = 4;
        config.gate_spacing_m = 1000.0;
        config.max_range_m = 10_000.0;
        config.ref_gate_texture = false;
        config.vel_gate_texture = false;
        config.terrain_blockage = false;
        let time = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let volume = try_build_synthetic_volume(&fields, time, &config).unwrap();
        let cut = &volume.cuts[0];
        let reflectivity = &cut.moments[&MomentType::Reflectivity];
        let velocity = &cut.moments[&MomentType::Velocity];
        assert!((reflectivity.scaled_value(1, 3).unwrap() - 40.0).abs() < 0.001);
        assert!(velocity.scaled_value(1, 3).unwrap() > 9.9);
        assert!(velocity.scaled_value(3, 3).unwrap() < -9.9);
        assert_eq!(volume.metadata.source_model.as_deref(), Some("WOOF"));
    }

    #[test]
    fn malformed_generic_volume_is_rejected_before_sampling() {
        let mut input = uniform_model();
        input.eastward_wind_mps.pop();
        let error = WrfRadarFields::from_model_fields(input).err().unwrap();
        assert!(error.contains("eastward wind has 7 samples; expected 8"));
    }
}
