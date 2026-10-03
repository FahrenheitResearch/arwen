//! Header-only admission prevents display frames from masquerading as columns.
use std::path::Path;

const MASS: &[&str] = &["bottom_top", "south_north", "west_east"];
const PLANE: &[&str] = &["south_north", "west_east"];
const INTERFACE: &[&str] = &["bottom_top_stag", "south_north", "west_east"];

pub fn validate_history(path: &Path) -> Result<(), String> {
    let file = netcrust::File::open(path).map_err(|e| e.to_string())?;
    if file.attribute("schema").and_then(|a| a.as_string().map(str::to_owned)).as_deref()
        == Some("native-atmosphere.columns/v1")
    {
        return Err("radar_input_needs_adapter: native-atmosphere.columns/v1 requires --canonical-atmosphere or the simulated-radar CLI's --input-kind native-columns before beam sampling".into());
    }
    let mut required: Vec<(&str, &[&str])> = vec![
        ("XLAT", PLANE), ("XLONG", PLANE), ("HGT", PLANE),
        ("T", &["Time", "bottom_top", "south_north", "west_east"]),
        ("PH", INTERFACE), ("PHB", INTERFACE), ("W", INTERFACE),
        ("U", &["bottom_top", "south_north", "west_east_stag"]),
        ("V", &["bottom_top", "south_north_stag", "west_east"]),
    ];
    if file.variable("REFL_10CM").is_some() {
        required.push(("REFL_10CM", MASS));
    } else {
        required.extend(["P", "PB", "QVAPOR", "QRAIN"].map(|name| (name, MASS)));
    }
    if file.attribute("RADAR_NATIVE_WINDS").and_then(|a| a.as_string().map(str::to_owned)).as_deref()
        == Some("earth-relative-mass-grid/v1")
    {
        required.extend([("RADAR_U_EARTH", MASS), ("RADAR_V_EARTH", MASS)]);
    }
    let missing: Vec<_> = required.iter().filter_map(|(name, _)|
        file.variable(name).is_none().then_some(*name)).collect();
    let dimensions: Vec<_> = MASS.iter().filter_map(|name|
        file.dimension(name).filter(|d| d.len() > 0).is_none().then_some(*name)).collect();
    if !missing.is_empty() || !dimensions.is_empty() {
        return Err(format!(
            "radar_input_missing_columns: {} lacks variables [{}] and positive dimensions [{}]; 2D display frames do not contain the vertical atmosphere needed for beam heights, reflectivity and radial velocity; supply full WRF history or the native-atmosphere.columns/v1 adapter",
            path.file_name().unwrap_or_default().to_string_lossy(), missing.join(", "), dimensions.join(", ")
        ));
    }
    for (name, expected) in required {
        let var = file.variable(name).expect("required fields checked");
        let mut actual: Vec<_> = var.dimensions().iter().map(|d| d.name()).collect();
        if name != "T" && actual.first() == Some(&"Time") {
            actual.remove(0);
        }
        if actual != expected || var.shape().contains(&0) {
            let time = if name == "T" { "; nonempty Time is required by the native grid reader" }
                       else { ", optionally preceded by nonempty Time" };
            return Err(format!(
                "radar_input_invalid_columns: {name} dimensions {actual:?} must be {expected:?}{time}; a surface or collapsed field cannot represent the vertical atmosphere"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn file_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("radar-input-{}-{}.nc", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)))
    }

    #[test]
    fn display_only_frame_names_all_missing_columns() {
        let path = file_path();
        let mut schema = Schema::new(NcFormat::Offset64);
        let y = schema.def_dim("south_north", 2, false).unwrap();
        let x = schema.def_dim("west_east", 2, false).unwrap();
        let mut variables = Vec::new();
        for name in ["XLAT", "XLONG", "HGT", "REFC", "U10", "V10"] {
            variables.push(schema.def_var(name, NcType::Float, &[y, x]).unwrap());
        }
        let mut writer = NcWriter::create(&path, schema).unwrap();
        for variable in variables {
            writer.write_var(variable, VarData::F32(&[0.0; 4])).unwrap();
        }
        writer.finish().unwrap();
        let error = validate_history(&path).unwrap_err();
        assert!(error.starts_with("radar_input_missing_columns:"));
        for name in ["PH", "PHB", "U", "V", "W", "P", "PB", "T", "QVAPOR", "QRAIN", "bottom_top"] {
            assert!(error.contains(name), "missing field not named: {name}: {error}");
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn native_columns_require_the_shared_adapter() {
        let path = file_path();
        let mut schema = Schema::new(NcFormat::Offset64);
        schema.put_global_attr("schema", AttrValue::Text("native-atmosphere.columns/v1".into())).unwrap();
        let writer = NcWriter::create(&path, schema).unwrap();
        writer.finish().unwrap();
        assert!(validate_history(&path).unwrap_err().starts_with("radar_input_needs_adapter:"));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn full_columns_accept_time_records_and_reject_collapsed_vertical_wind() {
        for collapse in [false, true] {
            let path = file_path();
            let mut schema = Schema::new(NcFormat::Offset64);
            let t = schema.def_dim("Time", 2, false).unwrap();
            let z = schema.def_dim("bottom_top", 2, false).unwrap();
            let zs = schema.def_dim("bottom_top_stag", 3, false).unwrap();
            let y = schema.def_dim("south_north", 2, false).unwrap();
            let ys = schema.def_dim("south_north_stag", 3, false).unwrap();
            let x = schema.def_dim("west_east", 2, false).unwrap();
            let xs = schema.def_dim("west_east_stag", 3, false).unwrap();
            let mut variables = Vec::new();
            for name in ["XLAT", "XLONG", "HGT"] {
                variables.push((schema.def_var(name, NcType::Float, &[t, y, x]).unwrap(), 8));
            }
            for name in ["PH", "PHB", "W"] {
                let dims = if collapse && name == "W" { vec![t, y, x] } else { vec![t, zs, y, x] };
                let elements = if collapse && name == "W" { 8 } else { 24 };
                variables.push((schema.def_var(name, NcType::Float, &dims).unwrap(), elements));
            }
            variables.push((schema.def_var("U", NcType::Float, &[t, z, y, xs]).unwrap(), 24));
            variables.push((schema.def_var("V", NcType::Float, &[t, z, ys, x]).unwrap(), 24));
            variables.push((schema.def_var("T", NcType::Float, &[t, z, y, x]).unwrap(), 16));
            variables.push((schema.def_var("REFL_10CM", NcType::Float, &[t, z, y, x]).unwrap(), 16));
            let mut writer = NcWriter::create(&path, schema).unwrap();
            for (variable, elements) in variables {
                writer.write_var(variable, VarData::F32(&vec![0.0; elements])).unwrap();
            }
            writer.finish().unwrap();
            if collapse {
                assert!(validate_history(&path).unwrap_err().contains("radar_input_invalid_columns: W"));
            } else {
                validate_history(&path).unwrap();
                bowecho_simradar::WrfFile::open(&path).unwrap();
            }
            std::fs::remove_file(path).unwrap();
        }
    }
}
