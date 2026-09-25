//! Recover a witnessed source-layer quantity error before WRF's vertical interpolation.
//! Original files are read-only; every changed column must reproduce the received
//! cold-initialization SMOIS before the source quantity is converted.
use std::{fs, io::Write, path::Path};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Clone, Deserialize)]
struct Source {
    #[serde(default = "soil_m")] source_variable: String,
    #[serde(default = "soil_levels")] source_depth_variable: String,
    source_quantity: String,
    source_units: String,
    #[serde(default)] source_layer_depths_m: Option<Vec<f64>>,
    #[serde(default)] source_depths_from_metgrid: bool,
    source_layer_bounds_m: Vec<[f64; 2]>,
}
fn soil_m() -> String { "SOILM".into() }
fn soil_levels() -> String { "SOIL_LEVELS".into() }

#[derive(Deserialize)]
struct Authority {
    schema: String,
    #[serde(flatten)] source: Source,
    #[serde(default)] liquid: Option<Source>,
}

struct Array { values: Vec<f64>, shape: Vec<usize>, dimensions: Vec<String> }

fn read(file: &netcrust::File, name: &str) -> Result<Array, String> {
    let variable = file.variable(name).ok_or_else(|| format!("missing required variable {name}"))?;
    let array = file.read_array_f64(name).map_err(|error| format!("cannot read {name}: {error}"))?;
    let shape = array.shape().to_vec();
    let mut values = array.into_values();
    let number = |key| variable.attribute(key).and_then(|attribute| attribute.as_f64());
    let scale = number("scale_factor").unwrap_or(1.0);
    let offset = number("add_offset").unwrap_or(0.0);
    if !scale.is_finite() || !offset.is_finite() || scale == 0.0 { return Err(format!("{name}: invalid CF packing")); }
    for value in &mut values {
        if !value.is_finite() || number("_FillValue").is_some_and(|fill| *value == fill)
            || number("missing_value").is_some_and(|fill| *value == fill) { *value = f64::NAN; }
        else { *value = *value * scale + offset; }
    }
    Ok(Array { values, shape, dimensions: variable.dimensions().iter().map(|dim|dim.name().to_string()).collect() })
}

fn text(file: &netcrust::File, name: &str) -> Option<String> {
    file.attribute(name).and_then(|value|value.as_string().map(str::to_string))
}

fn units(file: &netcrust::File, name: &str) -> String {
    file.variable(name).and_then(|var|var.attribute("units").and_then(|a|a.as_string()).map(str::to_string)).unwrap_or_default()
}

fn time(file: &netcrust::File) -> Result<String, String> {
    let variable = file.variable("Times").ok_or("missing WRF Times")?;
    if variable.shape() != [1, 19] { return Err("soil recovery requires exactly one WRF time record".into()); }
    String::from_utf8(super::character_bytes(file, &variable)?).map_err(|e|e.to_string())
}

fn geometry(array: &Array) -> String {
    array.dimensions.iter().zip(&array.shape)
        .map(|(name,size)| format!("{name}={size}")).collect::<Vec<_>>().join(", ")
}

/// [Time=1, soil layer, south_north, west_east], and for a SOURCE plane
/// also WRF's staggered spelling of the two horizontal names. Whether
/// such a file really carries the mass values in its leading block is
/// settled by its own coordinates in `recover`, never by a dimension
/// name; the target's soil field is a mass field by construction, so
/// `staggered` is false for it.
fn field_shape(array: &Array, label: &str, staggered: bool) -> Result<(usize,usize,usize),String> {
    let named = array.shape.len() == 4 && array.dimensions.len() == 4
        && array.shape[0] == 1 && array.dimensions[0] == "Time"
        && ["south_north","west_east"].iter().zip(&array.dimensions[2..]).all(|(plain,actual)|
            actual == plain || (staggered && *actual == format!("{plain}_stag")));
    if !named {
        return Err(format!("{label}: expected [Time=1, soil layer, south_north, west_east], found [{}]", geometry(array)));
    }
    Ok((array.shape[1], array.shape[2], array.shape[3]))
}

/// The leading `ny` by `nx` block of every layer of a `sy` by `sx` plane.
fn leading_block(values: &[f64], layers: usize, (sy,sx): (usize,usize), (ny,nx): (usize,usize)) -> Vec<f64> {
    if (sy,sx) == (ny,nx) { return values.to_vec(); }
    let mut kept = Vec::with_capacity(layers*ny*nx);
    for layer in 0..layers {
        for row in 0..ny {
            let start = layer*sy*sx + row*sx;
            kept.extend_from_slice(&values[start..start+nx]);
        }
    }
    kept
}

fn depth_list(values: &[f64]) -> String {
    values.iter().map(|value| format!("{value}")).collect::<Vec<_>>().join(", ")
}

fn bounds_list(bounds: &[[f64;2]]) -> String {
    bounds.iter().map(|[top,bottom]| format!("[{top}, {bottom}]")).collect::<Vec<_>>().join(", ")
}

/// Which authority layer declares each stacked source depth, in order.
///
/// A producing table lists every soil layer its source can carry and
/// metgrid stacks the ones that cycle actually contained, so the
/// authority is allowed to name more layers than the file holds. Layer
/// bounds are ordered and do not overlap (`validate_geometry`), so a
/// depth falls inside at most one of them: the pairing is read off the
/// file instead of being assumed to be position for position, and a
/// depth that no declared layer contains, or one that would pair out of
/// order, is refused with both lists rather than converted on the wrong
/// layer thickness.
fn declared_layers(depths_cm: &[f64], bounds: &[[f64;2]]) -> Result<Vec<usize>,String> {
    let mut mapping: Vec<usize> = Vec::with_capacity(depths_cm.len());
    for &centimeters in depths_cm {
        let depth = centimeters * 0.01;
        let found = bounds.iter().position(|[top,bottom]| depth >= top-1e-10 && depth <= bottom+1e-10);
        let ordered = match (found, mapping.last()) {
            (Some(index), None) => Some(index),
            (Some(index), Some(&previous)) if index > previous => Some(index),
            _ => None,
        };
        match ordered {
            Some(index) => mapping.push(index),
            None => return Err(format!(
                "the source stacks {} soil layer(s) at depths (cm) [{}] and the authority declares {} layer(s) with bounds (m) [{}]; every source depth must fall inside one declared layer, and a later depth inside a later layer",
                depths_cm.len(), depth_list(depths_cm), bounds.len(), bounds_list(bounds))),
        }
    }
    Ok(mapping)
}

fn close(got: f64, expected: f64) -> bool {
    got.is_finite() && expected.is_finite() && (got-expected).abs() <= 1e-6 * expected.abs().max(1.0)
}

fn quantity_scale(source: &Source) -> Result<Option<f64>, String> {
    let compact = source.source_units.to_lowercase().replace([' ', '^', '*', '(', ')'], "");
    match (source.source_quantity.as_str(), compact.as_str()) {
        ("layer_water_mass", "kgm-2" | "kg/m2") => Ok(Some(0.001)),
        ("equivalent_water_depth", "mm") => Ok(Some(0.001)),
        ("equivalent_water_depth", "m") => Ok(Some(1.0)),
        ("volume_fraction", "m3m-3" | "m3/m3" | "1") => Ok(None),
        _ => Err(format!("{}: quantity {} and units {:?} do not identify a supported layer water quantity", source.source_variable, source.source_quantity, source.source_units)),
    }
}

fn validate_geometry(source: &Source) -> Result<(), String> {
    let depths = source.source_layer_depths_m.as_ref();
    if source.source_depths_from_metgrid == depths.is_some() {
        return Err("soil authority must choose exactly one of explicit source_layer_depths_m or source_depths_from_metgrid".into());
    }
    if source.source_layer_bounds_m.len() < 2 || depths.is_some_and(|values|values.len() != source.source_layer_bounds_m.len()) {
        return Err("soil authority needs layer bounds and matching explicit depths when supplied for at least two layers".into());
    }
    for (index, &[top, bottom]) in source.source_layer_bounds_m.iter().enumerate() {
        let invalid_depth = depths.is_some_and(|values| !values[index].is_finite() || values[index] <= 0.0
            || values[index] < top || values[index] > bottom || (index > 0 && values[index] <= values[index-1]));
        if ![top,bottom].iter().all(|v|v.is_finite()) || top < 0.0 || bottom <= top || invalid_depth
            || (index > 0 && top < source.source_layer_bounds_m[index-1][1] - 1e-10) {
            return Err(format!("source layer {index}: finite ordered depths and nonoverlapping positive layer bounds are required"));
        }
    }
    quantity_scale(source)?;
    Ok(())
}

struct Prepared { raw: Vec<f64>, normalized: Vec<f64>, depths: Vec<f32>, layers: usize, declared: Vec<usize> }

fn prepare(file: &netcrust::File, authority: &Source, (ny,nx): (usize,usize), pad: (usize,usize), land: &[bool]) -> Result<Prepared,String> {
    validate_geometry(authority)?;
    let source = read(file, &authority.source_variable)?;
    let (layers, sy, sx) = field_shape(&source, &authority.source_variable, true)?;
    let extent = (ny+pad.0, nx+pad.1);
    if (sy,sx) != extent {
        return Err(format!("{}: the source soil plane covers {sy} by {sx} points while this file's own horizontal coordinates cover {} by {} for the {ny} by {nx} target domain",
            authority.source_variable, extent.0, extent.1));
    }
    let bounds = &authority.source_layer_bounds_m;
    if layers > bounds.len() {
        return Err(format!("{}: the source stacks {layers} soil layer(s) and the authority declares only {} layer(s) with bounds (m) [{}]",
            authority.source_variable, bounds.len(), bounds_list(bounds)));
    }
    let axis = read(file, &authority.source_depth_variable)?;
    // WRF's flagged SOIL_LEVELS input is in integer centimetres. Its empty
    // metgrid units attribute does not change that documented producer contract.
    let axis_units = units(file, &authority.source_depth_variable).trim().to_lowercase();
    if !matches!(axis_units.as_str(), "" | "cm" | "centimeters" | "centimetres") { return Err("WRF SOIL_LEVELS axis must carry centimetres (or its ordinary empty units attribute)".into()); }
    let cells = ny*nx;
    let broadcast_axis = axis.values.len() == layers && matches!(axis.shape.as_slice(), [_] | [1,_]);
    if !broadcast_axis && axis.shape != source.shape { return Err("source depth axis must be a layer vector or match the source moisture grid".into()); }
    let source_values = leading_block(&source.values, layers, (sy,sx), (ny,nx));
    let axis_values = if broadcast_axis { axis.values.clone() }
                      else { leading_block(&axis.values, layers, (sy,sx), (ny,nx)) };
    let first = land.iter().position(|v|*v).ok_or("no land columns to recover")?;
    let mut order = (0..layers).collect::<Vec<_>>();
    let level = |layer:usize, cell:usize| if broadcast_axis { axis_values[layer] } else { axis_values[layer*cells+cell] };
    order.sort_by(|&a,&b| level(a,first).total_cmp(&level(b,first)));
    let stacked = order.iter().map(|&original| level(original, first)).collect::<Vec<_>>();
    let declared = declared_layers(&stacked, bounds)?;
    let scale = quantity_scale(authority)?;
    let mut raw = vec![f64::NAN; layers*cells];
    let mut normalized = raw.clone();
    let mut depths = Vec::new();
    for (sorted, &original) in order.iter().enumerate() {
        let centimeters = level(original, first);
        let actual_depth = centimeters * 0.01;
        let [top,bottom] = bounds[declared[sorted]];
        if !centimeters.is_finite() || centimeters <= 0.0 || (centimeters-centimeters.round()).abs() > 1e-4
            || actual_depth < top-1e-10 || actual_depth > bottom+1e-10
            || authority.source_layer_depths_m.as_ref().is_some_and(|values|(actual_depth-values[declared[sorted]]).abs() > 1e-6)
            || depths.last().is_some_and(|previous| centimeters as f32 / 100.0 <= *previous) {
            return Err(format!("{}: the stacked source depth {centimeters} cm does not match declared layer {} with bounds (m) [{top}, {bottom}]",
                authority.source_depth_variable, declared[sorted]));
        }
        depths.push(centimeters as f32 / 100.0);
        let thickness = bottom-top;
        for (cell,&is_land) in land.iter().enumerate() {
            if !is_land { continue; }
            if !close(level(original,cell), centimeters) { return Err("source soil depths vary across the land domain".into()); }
            let value = source_values[original*cells+cell];
            if !value.is_finite() || value < 0.0 { return Err(format!("{}: missing or negative source water on land", authority.source_variable)); }
            let volume = scale.map_or(value, |scale| value*scale/thickness);
            // Naming the value, the layer and the thickness is the whole
            // of the diagnosis: the conversion is value*scale/thickness, so
            // a fraction outside [0,1] is either a source value that is not
            // the declared quantity or a declared layer whose bounds are
            // the wrong thickness for it, and those two are told apart by
            // the three numbers below and nothing else.
            if !volume.is_finite() || !(0.0..=1.0+1e-7).contains(&volume) { return Err(format!("{}: source-layer conversion produces an invalid volume fraction {volume} at authority layer index {} of bounds (m) [{top}, {bottom}], thickness {thickness} m, from source value {value}", authority.source_variable, declared[sorted])); }
            raw[sorted*cells+cell] = value;
            normalized[sorted*cells+cell] = volume;
        }
    }
    Ok(Prepared { raw, normalized, depths, layers, declared })
}

fn interpolate(values: &[f64], depths: &[f32], target: f32, cell: usize, cells: usize, floor_zero: bool) -> Result<f64,String> {
    let pair = depths.windows(2).position(|z| target >= z[0] && target <= z[1])
        .ok_or("target soil centre is outside the source interpolation depths")?;
    // Match WRF REAL's single-precision order, including its Noah-MP floor
    // for nonpositive interpolated total water. This is not a target-depth division.
    let value = ((values[pair*cells+cell] as f32 * (depths[pair+1]-target))
        + (values[(pair+1)*cells+cell] as f32 * (target-depths[pair]))) / (depths[pair+1]-depths[pair]);
    Ok(if floor_zero && value <= 0.0 { 0.005 } else { f64::from(value) })
}

fn recover_field(prepared: &Prepared, target: &Array, centers: &[f32], land: &[bool], floor_zero: bool) -> Result<(Vec<f64>,f64),String> {
    let cells = land.len();
    let mut result = target.values.clone();
    let mut max_error: f64 = 0.0;
    for (layer,&center) in centers.iter().enumerate() {
        for (cell,&is_land) in land.iter().enumerate() {
            if !is_land { continue; }
            let expected = interpolate(&prepared.raw,&prepared.depths,center,cell,cells,floor_zero)?;
            let received = target.values[layer*cells+cell];
            if !close(received,expected) { return Err(format!("raw source reconstruction does not reproduce the received soil field at layer {layer}, land cell {cell}: received {received}, reconstructed {expected}")); }
            max_error = max_error.max((received-expected).abs());
            result[layer*cells+cell] = interpolate(&prepared.normalized,&prepared.depths,center,cell,cells,floor_zero)?;
        }
    }
    Ok((result,max_error))
}

fn write_field(out: &Path, name: &str, values: &[f64], target: &Array) -> Result<Value,String> {
    let filename = format!("{name}.f64");
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(out.join(&filename)).map_err(|e|e.to_string())?;
    for value in values { file.write_all(&value.to_le_bytes()).map_err(|e|e.to_string())?; }
    Ok(json!({"name":name,"filename":filename,"shape":target.shape,"dimensions":target.dimensions,"dtype":"<f8","units":"m3 m-3"}))
}

pub(crate) fn recover(wrf_path: &Path, met_path: &Path, authority_path: &Path, output: &Path) -> Result<(),String> {
    let authority_json: Value = serde_json::from_slice(&fs::read(authority_path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let authority: Authority = serde_json::from_value(authority_json.clone()).map_err(|e|format!("invalid source soil authority: {e}"))?;
    if authority.schema != "gpuwm-wrf-soil-authority-v1" { return Err("unsupported soil authority schema".into()); }
    let (wrf, _) = super::open(wrf_path)?;
    let (met, _) = super::open(met_path)?;
    let valid_time = time(&wrf)?;
    if !text(&wrf,"TITLE").is_some_and(|title|title.contains("OUTPUT FROM REAL_EM"))
        || text(&wrf,"START_DATE").as_deref() != Some(valid_time.as_str())
        || text(&wrf,"SIMULATION_START_DATE").as_deref() != Some(valid_time.as_str()) {
        return Err("source soil recovery applies only to a cold REAL_EM initial state, never a restart".into());
    }
    if time(&met)? != valid_time { return Err("met_em and wrfinput valid times do not match".into()); }
    if met.attribute("FLAG_SOIL_LEVELS").and_then(|a|a.as_f64()) != Some(1.0) { return Err("source recovery requires WRF's explicit SOIL_LEVELS interpolation contract".into()); }
    let physics = wrf.attribute("SF_SURFACE_PHYSICS").and_then(|a|a.as_f64()).ok_or("missing target land-surface physics identity")?;
    if physics != 2.0 && physics != 4.0 { return Err("source soil recovery currently implements the witnessed Noah/Noah-MP REAL interpolation owner".into()); }
    let moisture = read(&wrf,"SMOIS")?;
    let (target_layers,ny,nx) = field_shape(&moisture,"SMOIS",false)?;
    let cells = nx.checked_mul(ny).ok_or("soil domain shape overflow")?;
    let mask = read(&wrf,"LANDMASK")?;
    if mask.shape != [1,ny,nx] || mask.values.iter().any(|v|!v.is_finite() || (*v != 0.0 && *v != 1.0)) { return Err("target land mask is missing or does not match the soil domain".into()); }
    let land = mask.values.iter().map(|v|*v > 0.5).collect::<Vec<_>>();
    if !moisture.values.iter().enumerate().any(|(index,value)|land[index%cells] && *value > 1.0) {
        return Err("land soil moisture does not exceed volume-fraction bounds; source recovery is unnecessary and would risk converting already-normalized WPS data twice".into());
    }
    // The source extent the file itself declares, and the proof that
    // its leading block is this domain. A plane written straight out of
    // WRF's own arrays carries the extra staggered row and column past
    // the mass points; one row or column of slack is admitted, and only
    // when the leading block of BOTH coordinates is this domain to the
    // bit. Anything else is named with both extents instead of being
    // cropped on a guess.
    let mut pad: Option<(usize,usize)> = None;
    for (target_name, source_name) in [("XLAT","XLAT_M"),("XLONG","XLONG_M")] {
        let left = read(&wrf,target_name)?; let right = read(&met,source_name)?;
        if left.shape != [1,ny,nx] {
            return Err(format!("target {target_name} is [{}] and the soil domain is [Time=1, {ny}, {nx}]", geometry(&left)));
        }
        let [sy,sx] = match right.shape.as_slice() { [1,y,x] => [*y,*x],
            _ => return Err(format!("{source_name} is [{}] and the target domain is [Time=1, {ny}, {nx}]", geometry(&right))) };
        if sy < ny || sx < nx || sy-ny > 1 || sx-nx > 1 {
            return Err(format!("{source_name} covers {sy} by {sx} points and the target {target_name} covers {ny} by {nx}; a source may carry WRF's one extra staggered row and column past the mass points, nothing else"));
        }
        let extra = (sy-ny, sx-nx);
        if pad.is_some_and(|already| already != extra) {
            return Err("XLAT_M and XLONG_M do not agree on the source horizontal extent".into());
        }
        pad = Some(extra);
        let same = (0..ny).all(|row| (0..nx).all(|column| {
            let (a,b) = (left.values[row*nx+column], right.values[row*sx+column]);
            a.is_finite() && b.is_finite() && (a as f32).to_bits() == (b as f32).to_bits()
        }));
        if !same { return Err(format!("source and target horizontal coordinates differ: {source_name}/{target_name}")); }
    }
    let pad = pad.unwrap_or((0,0));
    let centers = read(&wrf,"ZS")?;
    if centers.values.len() != target_layers || !matches!(centers.shape.as_slice(),[_] | [1,_])
        || centers.values.iter().any(|v|!v.is_finite() || *v <= 0.0)
        || centers.values.windows(2).any(|z|z[0] >= z[1]) { return Err("target ZS must be a finite ordered soil-centre vector".into()); }
    if units(&wrf,"ZS").trim() != "m" { return Err("target ZS must declare metre depths".into()); }
    let centers = centers.values.iter().map(|v|*v as f32).collect::<Vec<_>>();
    let prepared = prepare(&met,&authority.source,(ny,nx),pad,&land)?;
    let (corrected,max_error) = recover_field(&prepared,&moisture,&centers,&land,physics == 4.0)?;
    let mut liquid_values = None;
    let liquid_action;
    if let Some(liquid_source) = &authority.liquid {
        let liquid = read(&wrf,"SH2O")?;
        if liquid.shape != moisture.shape { return Err("SH2O shape differs from SMOIS".into()); }
        let source = prepare(&met,liquid_source,(ny,nx),pad,&land)?;
        let (values,_) = recover_field(&source,&liquid,&centers,&land,false)?;
        liquid_values = Some(values);
        liquid_action = "recovered_from_separate_source_authority";
    } else if wrf.variable("SH2O").is_some() {
        let liquid = read(&wrf,"SH2O")?;
        if liquid.shape != moisture.shape { return Err("SH2O shape differs from SMOIS".into()); }
        liquid_action = if liquid.values.iter().enumerate().all(|(i,v)|!land[i%cells] || *v == 0.0) {
            "preserved_zero_cold_initialization_placeholder"
        } else { "preserved_existing_target_liquid" };
        for (index,&value) in liquid.values.iter().enumerate() {
            if land[index%cells] && (!value.is_finite() || value < 0.0 || value > corrected[index]+1e-7) {
                return Err("target liquid water is incompatible with corrected total water; a separate source liquid-water authority is required".into());
            }
        }
    } else { liquid_action = "no_target_liquid_field"; }
    if let Some(liquid) = &liquid_values {
        for (index,&value) in liquid.iter().enumerate() {
            if land[index%cells] && (!value.is_finite() || value < 0.0 || value > corrected[index]+1e-7) { return Err("recovered liquid water exceeds corrected total water".into()); }
        }
    }
    // Finish all causal and physical checks before creating any output.
    fs::create_dir_all(output).map_err(|e|e.to_string())?;
    let mut variables = vec![write_field(output,"SMOIS",&corrected,&moisture)?];
    if let Some(values) = &liquid_values { variables.push(write_field(output,"SH2O",values,&moisture)?); }
    let receipt = json!({"operation":"source_quantity_conversion_before_wrf_real_soil_interpolation",
        "valid_time":valid_time,"cold_initialization":true,"source_layers":prepared.layers,
        "target_layers":target_layers,"land_columns":land.iter().filter(|v|**v).count(),
        "raw_forward_reconstruction_max_abs_error":max_error,"liquid_action":liquid_action,
        "source_interpolation_depths_m":prepared.depths,
        "authority_layers_used":prepared.declared,
        "source_grid_extra_row_and_column":[pad.0,pad.1],
        "authority":authority_json,"original_files_modified":false,
        "source_wrfinput":wrf_path,"source_met_em":met_path,
        "water_columns_preserved":land.iter().filter(|v|!**v).count()});
    let metadata = json!({"schema":"gpuwm-wrf-soil-recovery-v1","variables":variables,"receipt":receipt});
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(output.join("metadata.json")).map_err(|e|e.to_string())?;
    file.write_all(&serde_json::to_vec_pretty(&metadata).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_layer_conversion_precedes_real_interpolation() {
        let raw = vec![1.,3.,12.,45.,162.,567.,1944.,6561.];
        let thickness = [0.01,0.02,0.06,0.18,0.54,1.62,4.86,14.58];
        let depths = [0.01,0.02,0.06,0.18,0.54,1.62,4.86,14.58];
        let volume = raw.iter().zip(thickness).map(|(value,dz)|value*0.001/dz).collect::<Vec<_>>();
        let target = [0.05,0.25,0.7,1.5];
        let expected = [0.1875,0.25972223,0.30740744,0.34444448];
        for ((z,correct),dz) in target.into_iter().zip(expected).zip([0.1,0.3,0.6,1.]) {
            let restored = interpolate(&volume,&depths,z,0,1,false).unwrap();
            assert!(close(restored,correct));
            let naive = interpolate(&raw,&depths,z,0,1,false).unwrap()*0.001/dz;
            assert!((naive-restored).abs() > 0.01);
        }
    }
    #[test]
    fn reconstruction_and_domain_depth_coverage_are_required() {
        let source = Prepared {raw:vec![10.,20.],normalized:vec![0.1,0.2],depths:vec![0.1,0.3],layers:2,declared:vec![0,1]};
        let target = Array {values:vec![17.],shape:vec![1,1,1,1],dimensions:Vec::new()};
        assert!(recover_field(&source,&target,&[0.2],&[true],false).is_err());
        assert!(interpolate(&source.raw,&source.depths,0.4,0,1,false).is_err());
    }
    #[test]
    fn declared_bounds_and_actual_axis_are_an_explicit_exclusive_authority_mode() {
        let value = json!({"source_quantity":"layer_water_mass","source_units":"kg m-2",
            "source_depths_from_metgrid":true,"source_layer_bounds_m":[[0.,0.01],[0.01,0.03]]});
        let mut source: Source = serde_json::from_value(value).unwrap();
        validate_geometry(&source).unwrap();
        source.source_layer_depths_m = Some(vec![0.01,0.02]);
        assert!(validate_geometry(&source).is_err());
        source.source_depths_from_metgrid = false;
        validate_geometry(&source).unwrap();
    }

    #[test]
    fn a_table_that_names_more_layers_than_the_file_stacks_pairs_by_depth() {
        // Nine declared layers, four stacked: the pairing is read off the
        // depths, so each layer converts on its own declared thickness.
        let bounds = [[0.,0.01],[0.01,0.04],[0.04,0.1],[0.1,0.3],[0.3,0.6],
                      [0.6,1.0],[1.0,1.6],[1.6,3.0],[3.0,10.0]];
        assert_eq!(declared_layers(&[1.,4.,30.,100.], &bounds).unwrap(), vec![0,1,3,5]);
        assert_eq!(declared_layers(&[1.,4.,10.,30.], &bounds[..4]).unwrap(), vec![0,1,2,3]);
    }

    #[test]
    fn a_depth_no_declared_layer_holds_is_refused_with_both_lists() {
        let bounds = [[0.,0.01],[0.01,0.04]];
        let refusal = declared_layers(&[1.,9.], &bounds).unwrap_err();
        assert!(refusal.contains("depths (cm) [1, 9]"), "{refusal}");
        assert!(refusal.contains("bounds (m) [[0, 0.01], [0.01, 0.04]]"), "{refusal}");
        // Two depths inside one declared layer would pair out of order.
        assert!(declared_layers(&[0.5,1.], &bounds).is_err());
        // A stack deeper than the declaration cannot pair at all.
        assert!(declared_layers(&[1.,4.,10.], &bounds).is_err());
    }

    #[test]
    fn a_staggered_source_plane_keeps_its_leading_mass_block() {
        // Two layers on a 3x3 staggered extent; the 2x2 mass block is
        // the leading row and column of each layer.
        let values: Vec<f64> = (0..18).map(f64::from).collect();
        assert_eq!(leading_block(&values, 2, (3,3), (2,2)),
                   vec![0.,1.,3.,4.,9.,10.,12.,13.]);
        assert_eq!(leading_block(&values, 2, (3,3), (3,3)), values);
    }
}
