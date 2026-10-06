//! Bind published invariant fields to a primary series without reconstructing them.
use std::{fs, path::{Path, PathBuf}};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use serde_json::json;

fn read(file: &netcrust::File, name: &str) -> Result<(Vec<f64>, Vec<usize>), String> {
    let var = file.variable(name);
    if var.is_none() && !file.has_hdf5_dataset(name) { return Err(format!("missing invariant coordinate or field {name}")); }
    let array = file.read_array_f64(name).map_err(|e| format!("cannot decode {name}: {e}"))?;
    let shape = array.shape().to_vec();
    let mut values = array.into_values();
    let number = |key| var.as_ref().and_then(|v|v.attribute(key)).and_then(|a|a.as_f64());
    let scale = number("scale_factor").unwrap_or(1.0);
    let offset = number("add_offset").unwrap_or(0.0);
    if !scale.is_finite() || scale == 0.0 || !offset.is_finite() { return Err(format!("{name}: invalid CF packing")); }
    for value in &mut values {
        if !value.is_finite() || number("_FillValue").is_some_and(|v|*value == v)
            || number("missing_value").is_some_and(|v|*value == v) { *value = f64::NAN; }
        else { *value = *value * scale + offset; }
    }
    Ok((values, shape))
}

fn units(file: &netcrust::File, name: &str) -> String {
    file.variable(name).and_then(|v|v.attribute("units").and_then(|a|a.as_string()).map(str::to_owned))
        .or_else(||file.hdf5_dataset_attribute_string(name,"units")).unwrap_or_default()
}

fn coordinate_indices(source: &[f64], target: &[f64], cyclic: bool) -> Result<Vec<usize>, String> {
    target.iter().map(|value| {
        if !value.is_finite() { return Err("nonfinite primary coordinate".into()); }
        let matches: Vec<_> = source.iter().enumerate().filter(|(_,candidate)| {
            let delta = (*candidate - value).abs();
            let delta = if cyclic { (delta % 360.0).min(360.0 - delta % 360.0) } else { delta };
            delta < 1.0e-5
        }).map(|(index,_)|index).collect();
        match matches.as_slice() { [index] => Ok(*index), _ => Err(format!("primary coordinate {value} has {} exact invariant matches",matches.len())) }
    }).collect()
}

fn subset(path: &Path, name: &str, latitude: &[f64], longitude: &[f64], expected_units: &[&str]) -> Result<Vec<f64>, String> {
    let (file,_) = super::open(path)?;
    let received_units = units(&file,name);
    if !expected_units.contains(&received_units.as_str()) { return Err(format!("{name}: units {received_units:?}, expected {expected_units:?}")); }
    let (lat,_) = read(&file,"lat")?;
    let (lon,_) = read(&file,"lon")?;
    let rows = coordinate_indices(&lat,latitude,false)?;
    let columns = coordinate_indices(&lon,longitude,true)?;
    let (values,shape) = read(&file,name)?;
    let expected = vec![lat.len(),lon.len()];
    if shape != expected && shape != [1,lat.len(),lon.len()] { return Err(format!("{name}: published invariant must have one plane on lat/lon, got {shape:?}")); }
    let mut output = Vec::with_capacity(latitude.len()*longitude.len());
    for row in rows { for &column in &columns { output.push(values[row*lon.len()+column]); } }
    if !output.iter().all(|v|v.is_finite()) { return Err(format!("{name}: published invariant is missing on the requested grid")); }
    Ok(output)
}

pub(super) fn bind(height: &Path, height_var: &str, land: &Path, land_var: &str, output: &Path, series: &[PathBuf]) -> Result<(), String> {
    if output.exists() { return Err(format!("{} already exists; invariant binding never replaces it",output.display())); }
    let mut times = Vec::new();
    let mut latitude = Vec::new();
    let mut longitude = Vec::new();
    let mut time_units = String::new();
    for (index,path) in series.iter().enumerate() {
        let (file,_) = super::open(path)?;
        let (lat,lat_shape) = read(&file,"lat")?;
        let (lon,lon_shape) = read(&file,"lon")?;
        let (time,time_shape) = read(&file,"time")?;
        let received_units = units(&file,"time");
        if lat_shape != [lat.len()] || lon_shape != [lon.len()] || time_shape != [time.len()] || time.is_empty() { return Err("primary time/latitude/longitude coordinates must be nonempty 1D axes".into()); }
        if index == 0 { latitude = lat; longitude = lon; time_units = received_units; }
        else if latitude != lat || longitude != lon || time_units != received_units { return Err("primary series coordinates or time units change between annual files".into()); }
        times.extend(time);
    }
    if times.is_empty() || !times.iter().all(|v|v.is_finite()) || times.windows(2).any(|w|w[0] >= w[1]) { return Err("primary valid times must be finite and strictly increasing".into()); }
    let reference = super::parse_cf_units(&time_units)?.ok_or("primary time units are not CF reference-time units")?;
    let valid_times = super::decode_times(&reference,&times)?;
    let terrain = subset(height,height_var,&latitude,&longitude,&["m"])?;
    let mask = subset(land,land_var,&latitude,&longitude,&["1","frac."])?;
    if mask.iter().any(|v|*v < 0.0 || *v > 1.0) { return Err("published land fraction lies outside 0..1".into()); }
    let mut schema = Schema::new(NcFormat::Classic);
    let t = schema.def_dim("time",times.len(),false).map_err(|e|e.to_string())?;
    let y = schema.def_dim("lat",latitude.len(),false).map_err(|e|e.to_string())?;
    let x = schema.def_dim("lon",longitude.len(),false).map_err(|e|e.to_string())?;
    let mut variable = |name, dimensions: &[usize], units: &str, standard: &str| -> Result<usize,String> {
        let id = schema.def_var(name,NcType::Double,dimensions).map_err(|e|e.to_string())?;
        for (key,value) in [("units",units),("standard_name",standard)] { schema.put_var_attr(id,key,AttrValue::Text(value.into())).map_err(|e|e.to_string())?; }
        Ok(id)
    };
    let time_id = variable("time",&[t],&time_units,"time")?;
    let lat_id = variable("lat",&[y],"degrees_north","latitude")?;
    let lon_id = variable("lon",&[x],"degrees_east","longitude")?;
    let orog_id = variable("orog",&[t,y,x],"m","surface_altitude")?;
    let land_id = variable("land",&[t,y,x],"1","land_area_fraction")?;
    schema.put_var_attr(time_id,"calendar",AttrValue::Text("gregorian".into())).map_err(|e|e.to_string())?;
    for id in [orog_id,land_id] { schema.put_var_attr(id,"level_desc",AttrValue::Text("Surface".into())).map_err(|e|e.to_string())?; }
    let parent = output.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(|e|e.to_string())?;
    let temporary = output.with_extension(format!("nc.{}.part",std::process::id()));
    let result = (|| -> Result<(),String> {
        let mut writer = NcWriter::create(&temporary,schema).map_err(|e|e.to_string())?;
        for (id,values) in [(time_id,&times),(lat_id,&latitude),(lon_id,&longitude)] { writer.write_var(id,VarData::F64(values)).map_err(|e|e.to_string())?; }
        let terrain_all = terrain.repeat(times.len());
        let land_all = mask.repeat(times.len());
        writer.write_var(orog_id,VarData::F64(&terrain_all)).map_err(|e|e.to_string())?;
        writer.write_var(land_id,VarData::F64(&land_all)).map_err(|e|e.to_string())?;
        writer.finish().map_err(|e|e.to_string())?;
        fs::rename(&temporary,output).map_err(|e|e.to_string())
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result?;
    println!("{}",json!({"schema":"gpuwm-published-invariant-binding-v1","method":"published_invariants_exact_coordinate_subset_repeated_at_primary_valid_times","valid_times":valid_times,"time_count":times.len(),"latitude_count":latitude.len(),"longitude_count":longitude.len(),"terrain_min_m":terrain.iter().copied().fold(f64::INFINITY,f64::min),"terrain_max_m":terrain.iter().copied().fold(f64::NEG_INFINITY,f64::max),"land_fraction_min":mask.iter().copied().fold(f64::INFINITY,f64::min),"land_fraction_max":mask.iter().copied().fold(f64::NEG_INFINITY,f64::max)}));
    Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn exact_grid_binding_handles_longitude_conventions() {
        assert_eq!(coordinate_indices(&[260.0,261.0,262.0],&[-100.0,-99.0],true).unwrap(),vec![0,1]);
        assert!(coordinate_indices(&[260.0,261.0],&[-99.5],true).is_err());
        assert!(coordinate_indices(&[0.0,360.0],&[0.0],true).is_err());
    }

    fn fixture(path: &Path, name: &str, units: &str, latitude: &[f64], longitude: &[f64], times: &[f64], values: &[f64]) {
        let mut schema = Schema::new(NcFormat::Classic);
        let t = schema.def_dim("time",times.len(),false).unwrap();
        let y = schema.def_dim("lat",latitude.len(),false).unwrap();
        let x = schema.def_dim("lon",longitude.len(),false).unwrap();
        let time_id = schema.def_var("time",NcType::Double,&[t]).unwrap();
        schema.put_var_attr(time_id,"units",AttrValue::Text("hours since 1800-1-1 00:00:0.0".into())).unwrap();
        let lat_id = schema.def_var("lat",NcType::Double,&[y]).unwrap();
        let lon_id = schema.def_var("lon",NcType::Double,&[x]).unwrap();
        let field = schema.def_var(name,NcType::Double,&[t,y,x]).unwrap();
        schema.put_var_attr(field,"units",AttrValue::Text(units.into())).unwrap();
        let mut writer = NcWriter::create(path,schema).unwrap();
        for (id,data) in [(time_id,times),(lat_id,latitude),(lon_id,longitude),(field,values)] {
            writer.write_var(id,VarData::F64(data)).unwrap();
        }
        writer.finish().unwrap();
    }

    #[test] fn published_fields_survive_exact_subset_and_time_binding() {
        let root = std::env::temp_dir().join(format!("invariant-binding-{}",std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let height = root.join("height.nc");
        let land = root.join("land.nc");
        let primary = root.join("primary.nc");
        let output = root.join("output.nc");
        let _ = fs::remove_file(&output);
        fixture(&height,"hgt","m",&[40.0,39.0],&[260.0,261.0,262.0],&[315552.0],&[10.0,20.0,30.0,40.0,50.0,60.0]);
        fixture(&land,"land","frac.",&[40.0,39.0],&[260.0,261.0,262.0],&[315552.0],&[0.0,0.25,1.0,0.0,0.5,1.0]);
        fixture(&primary,"dummy","K",&[39.0,40.0],&[-99.0,-98.0],&[1097676.0,1097679.0],&[280.0;8]);
        bind(&height,"hgt",&land,"land",&output,&[primary.clone()]).unwrap();
        let (file,_) = super::super::open(&output).unwrap();
        assert_eq!(read(&file,"orog").unwrap().0,vec![50.0,60.0,20.0,30.0,50.0,60.0,20.0,30.0]);
        assert_eq!(read(&file,"land").unwrap().0,vec![0.5,1.0,0.25,1.0,0.5,1.0,0.25,1.0]);
        assert_eq!(read(&file,"time").unwrap().0,vec![1097676.0,1097679.0]);
        assert!(bind(&height,"hgt",&land,"land",&output,&[primary]).unwrap_err().contains("already exists"));
        fs::remove_dir_all(&root).unwrap();
    }
}
