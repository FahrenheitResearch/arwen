//! Canonical radar columns, independent of the model that produced them.
//! Transport is top-down Z/Y/X. Conversion and all disk data work stay native.
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use std::path::Path;

const SCHEMA: &str = "native-atmosphere.columns/v1";
const WINDS: &str = "earth-relative-mass-grid/v1";
const WRF_G: f64 = 9.80665;
const WRF_KAPPA: f64 = 0.2857142857;
pub const ABI: &str =
    "native-atmosphere.columns/v1 temperature=temperature_k winds=earth-relative-mass-grid/v1";
type Result<T> = std::result::Result<T, String>;

fn text(file: &netcrust::File, name: &str) -> Result<String> {
    file.attribute(name)
        .and_then(|a| a.as_string().map(str::to_owned))
        .ok_or_else(|| format!("native atmosphere requires text attribute {name}"))
}
fn read(file: &netcrust::File, name: &str, shape: &[usize]) -> Result<Vec<f64>> {
    let var = file
        .variable(name)
        .ok_or_else(|| format!("native atmosphere lacks {name}"))?;
    if var.shape() != shape {
        return Err(format!(
            "native atmosphere {name} shape {:?} differs from {shape:?}",
            var.shape()
        ));
    }
    let values = file.read_f64(name).map_err(|e| e.to_string())?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "native atmosphere {name} contains nonfinite values"
        ));
    }
    Ok(values)
}
fn flip(values: &[f64], nz: usize, plane: usize) -> Vec<f64> {
    (0..nz)
        .rev()
        .flat_map(|k| values[k * plane..(k + 1) * plane].iter().copied())
        .collect()
}
fn faces(values: &[f64], nz: usize, ny: usize, nx: usize, x: bool) -> Vec<f64> {
    let (oy, ox) = if x { (ny, nx + 1) } else { (ny + 1, nx) };
    let mut out = Vec::with_capacity(nz * oy * ox);
    for k in 0..nz {
        for j in 0..oy {
            for i in 0..ox {
                let a = if x {
                    (j, i.saturating_sub(1))
                } else {
                    (j.saturating_sub(1), i)
                };
                let b = (j.min(ny - 1), i.min(nx - 1));
                out.push(
                    (values[(k * ny + a.0) * nx + a.1] + values[(k * ny + b.0) * nx + b.1]) * 0.5,
                );
            }
        }
    }
    out
}

/// Native mass winds remove the artificial second smoothing from a WRF
/// staggering round trip. Ordinary WRF files keep their own staggered winds.
pub fn native_winds(
    file: &bowecho_simradar::WrfFile,
    time: usize,
    fields: &mut bowecho_simradar::WrfRadarFields,
) -> Result<()> {
    if file.global_attr_str("RADAR_NATIVE_WINDS").ok().as_deref() != Some(WINDS) {
        return Ok(());
    }
    for (name, target) in [
        ("RADAR_U_EARTH", &mut fields.u),
        ("RADAR_V_EARTH", &mut fields.v),
    ] {
        let values = file.read_var(name, time).map_err(|e| e.to_string())?;
        // A bounded mesh window can have uncovered cells. Their NaNs carry
        // missing data through the same sampling mask as ordinary WRF winds.
        if values.len() != target.len()
            || values
                .iter()
                .any(|v| v.is_infinite() || (v.is_finite() && !(*v as f32).is_finite()))
        {
            return Err(format!(
                "{name} must match the mass grid without infinite or unrepresentable wind values"
            ));
        }
        *target = values.iter().map(|v| *v as f32).collect();
    }
    Ok(())
}

pub fn convert(source: &Path, destination: &Path) -> Result<()> {
    let file = netcrust::File::open(source).map_err(|e| e.to_string())?;
    if text(&file, "schema")? != SCHEMA {
        return Err("unsupported native atmosphere schema".into());
    }
    let dim = |name| {
        file.dimension(name)
            .map(|d| d.len())
            .filter(|n| *n >= 2)
            .ok_or_else(|| {
                format!("native atmosphere requires dimension {name} with at least two cells")
            })
    };
    let (nz, ny, nx) = (dim("level")?, dim("latitude")?, dim("longitude")?);
    let plane = ny
        .checked_mul(nx)
        .ok_or("native atmosphere grid size overflow")?;
    let budget = (nz as u64)
        .checked_add(1)
        .and_then(|n| n.checked_mul(plane as u64))
        .and_then(|n| n.checked_mul(96))
        .ok_or("native atmosphere memory estimate overflow")?;
    if rw_host_memory::available_bytes().is_some_and(|a| budget > a) {
        return Err(format!(
            "native atmosphere conversion requires {budget} available host bytes for its columns"
        ));
    }
    let valid = text(&file, "valid_time")?;
    let start = text(&file, "simulation_start")?;
    let parse = |s: &str| {
        chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d_%H:%M:%S").map_err(|e| e.to_string())
    };
    let elapsed = (parse(&valid)? - parse(&start)?).num_seconds();
    if elapsed < 0 {
        return Err("native atmosphere valid time precedes simulation start".into());
    }
    let lats = read(&file, "latitude_deg", &[ny])?;
    let lons = read(&file, "longitude_deg", &[nx])?;
    if lats.iter().any(|v| !(-90.0..=90.0).contains(v)) || lats.windows(2).any(|v| v[1] <= v[0]) {
        return Err(
            "native atmosphere latitude must increase south to north within [-90,90]".into(),
        );
    }
    if lons.windows(2).any(|v| v[1] <= v[0]) || lons[nx - 1] - lons[0] >= 360.0 {
        return Err("native atmosphere longitude must increase over a span smaller than 360 degrees without a duplicate seam".into());
    }
    let qv = flip(&read(&file, "qv_kg_kg", &[nz, ny, nx])?, nz, plane);
    if qv.iter().any(|v| !(0.0..1.0).contains(v)) {
        return Err("native vapor mass fraction must lie in [0,1) for dry-air conversion".into());
    }
    let native_temperature = file.variable("temperature_k").is_some();
    let temperature_pressure = if native_temperature {
        let values = flip(&read(&file, "pressure_pa", &[nz, ny, nx])?, nz, plane);
        if values.iter().any(|v| *v <= 0.0) {
            return Err("pressure must be positive".into());
        }
        Some(values)
    } else {
        None
    };
    let mut schema = Schema::new(NcFormat::Offset64);
    let mut dims = Vec::new();
    for (name, n, unlimited) in [
        ("Time", 0, true),
        ("DateStrLen", 19, false),
        ("bottom_top", nz, false),
        ("south_north", ny, false),
        ("west_east", nx, false),
        ("bottom_top_stag", nz + 1, false),
        ("south_north_stag", ny + 1, false),
        ("west_east_stag", nx + 1, false),
    ] {
        dims.push(
            schema
                .def_dim(name, n, unlimited)
                .map_err(|e| e.to_string())?,
        );
    }
    let mut attr = |name: &str, value: AttrValue| {
        schema
            .put_global_attr(name, value)
            .map_err(|e| e.to_string())
    };
    for (name, value) in [
        ("TITLE", "Native atmosphere radar scene".into()),
        ("START_DATE", start.clone()),
        ("SIMULATION_START_DATE", start),
        ("GPUWM_MODEL_LABEL", text(&file, "source_model")?),
        ("RADAR_COLUMNS_SCHEMA", SCHEMA.into()),
        ("RADAR_NATIVE_WINDS", WINDS.into()),
        ("RADAR_SOURCE_SHA256", crate::manifest::file_hash(source)?.0),
        (
            "RADAR_MOISTURE_CONVERSION",
            "source moist-air mass fraction divided by (1-qv); output dry-air mixing ratio".into(),
        ),
    ] {
        attr(name, AttrValue::Text(value))?;
    }
    for (src, dst) in [
        ("source_checkpoint", "RADAR_SOURCE_CHECKPOINT"),
        ("config_sha256", "RADAR_SOURCE_CONFIG_SHA256"),
        ("microphysics_scheme", "RADAR_MICROPHYSICS_SCHEME"),
        ("vertical_velocity_method", "RADAR_VERTICAL_VELOCITY_METHOD"),
        ("derivative_stencil", "RADAR_VERTICAL_VELOCITY_STENCIL"),
    ] {
        attr(dst, AttrValue::Text(text(&file, src)?))?;
    }
    for (src, dst) in [
        ("derivative_interval_s", "RADAR_DERIVATIVE_INTERVAL_S"),
        ("gravity_m_s2", "RADAR_SOURCE_GRAVITY_M_S2"),
    ] {
        let value = file
            .attribute(src)
            .and_then(|a| a.as_f64())
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| format!("native atmosphere lacks positive {src}"))?;
        attr(dst, AttrValue::Doubles(vec![value]))?;
    }
    let mp = file
        .attribute("mp_physics")
        .and_then(|a| a.as_f64())
        .filter(|v| v.is_finite() && *v >= 0.0 && v.fract() == 0.0 && *v <= i32::MAX as f64)
        .ok_or("native atmosphere lacks integer mp_physics")?;
    attr("MP_PHYSICS", AttrValue::Ints(vec![mp as i32]))?;
    if let Some(value) = file.attribute("morr_rimed_ice").and_then(|a| a.as_f64()) {
        if !value.is_finite()
            || value.fract() != 0.0
            || value < i32::MIN as f64
            || value > i32::MAX as f64
        {
            return Err("invalid morr_rimed_ice".into());
        }
        attr("MORR_RIMED_ICE", AttrValue::Ints(vec![value as i32]))?;
    }
    attr("MAP_PROJ", AttrValue::Ints(vec![6]))?;
    attr("RADAR_TEMPERATURE_BASIS", AttrValue::Text(if native_temperature {
        "native temperature_k; potential temperature encoded for the WRF reader"
    } else {
        "potential_temperature_k using the WRF reader reference of 100000 Pa and kappa 0.2857142857"
    }.into()))?;
    attr("GRID_ID", AttrValue::Ints(vec![1]))?;
    let spacing = (lats
        .windows(2)
        .map(|v| v[1] - v[0])
        .fold(f64::INFINITY, f64::min)
        * 111_195.0)
        .max(1.0);
    attr("DX", AttrValue::Doubles(vec![spacing]))?;
    attr("DY", AttrValue::Doubles(vec![spacing]))?;
    drop(attr);
    let times = schema
        .def_var("Times", NcType::Char, &[dims[0], dims[1]])
        .map_err(|e| e.to_string())?;
    // WRF field, input transport name, dimension ids, units.
    let mut plan: Vec<(&str, &str, Vec<usize>, &str)> = vec![
        ("XLAT", "", vec![0, 3, 4], "degree_north"),
        ("XLONG", "", vec![0, 3, 4], "degree_east"),
        ("HGT", "terrain_height_m", vec![0, 3, 4], "m"),
        ("SINALPHA", "", vec![0, 3, 4], "1"),
        ("COSALPHA", "", vec![0, 3, 4], "1"),
        (
            "T",
            if native_temperature {
                "temperature_k"
            } else {
                "potential_temperature_k"
            },
            vec![0, 2, 3, 4],
            "K",
        ),
        ("P", "pressure_pa", vec![0, 2, 3, 4], "Pa"),
        ("PB", "", vec![0, 2, 3, 4], "Pa"),
        ("PH", "", vec![0, 5, 3, 4], "m2 s-2"),
        ("PHB", "height_half_m", vec![0, 5, 3, 4], "m2 s-2"),
        ("W", "vertical_velocity_half_m_s", vec![0, 5, 3, 4], "m s-1"),
        ("U", "eastward_wind_m_s", vec![0, 2, 3, 7], "m s-1"),
        ("V", "northward_wind_m_s", vec![0, 2, 6, 4], "m s-1"),
        (
            "RADAR_U_EARTH",
            "eastward_wind_m_s",
            vec![0, 2, 3, 4],
            "m s-1",
        ),
        (
            "RADAR_V_EARTH",
            "northward_wind_m_s",
            vec![0, 2, 3, 4],
            "m s-1",
        ),
    ];
    for (src, dst) in [
        ("qv_kg_kg", "QVAPOR"),
        ("qc_kg_kg", "QCLOUD"),
        ("qr_kg_kg", "QRAIN"),
        ("qi_kg_kg", "QICE"),
        ("qs_kg_kg", "QSNOW"),
        ("qg_kg_kg", "QGRAUP"),
    ] {
        plan.push((dst, src, vec![0, 2, 3, 4], "kg kg-1"));
    }
    for (src, dst) in [
        ("nc_kg1", "QNCLOUD"),
        ("nr_kg1", "QNRAIN"),
        ("ni_kg1", "QNICE"),
        ("ns_kg1", "QNSNOW"),
        ("ng_kg1", "QNGRAUP"),
    ] {
        if file.variable(src).is_some() {
            plan.push((dst, src, vec![0, 2, 3, 4], "kg-1"));
        }
    }
    let mut ids = Vec::new();
    for (name, _, axes, units) in &plan {
        let id = schema
            .def_var(
                name,
                NcType::Double,
                &axes.iter().map(|i| dims[*i]).collect::<Vec<_>>(),
            )
            .map_err(|e| e.to_string())?;
        schema
            .put_var_attr(id, "units", AttrValue::Text((*units).into()))
            .map_err(|e| e.to_string())?;
        ids.push(id);
    }
    let parent = destination
        .parent()
        .ok_or("native atmosphere output needs a parent directory")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let tmp = destination.with_extension(format!("{}.partial", std::process::id()));
    let result = (|| {
        let mut writer = NcWriter::create(&tmp, schema).map_err(|e| e.to_string())?;
        writer
            .write_record(0, times, VarData::Char(valid.as_bytes()))
            .map_err(|e| e.to_string())?;
        for ((name, source, axes, _), id) in plan.iter().zip(ids) {
            let layers = if axes.contains(&5) {
                nz + 1
            } else if axes.contains(&2) {
                nz
            } else {
                1
            };
            let mut values = if source.is_empty() {
                vec![0.0; layers * plane]
            } else if layers == 1 {
                read(&file, source, &[ny, nx])?
            } else {
                flip(&read(&file, source, &[layers, ny, nx])?, layers, plane)
            };
            match *name {
                "XLAT" => {
                    for j in 0..ny {
                        values[j * nx..(j + 1) * nx].fill(lats[j]);
                    }
                }
                "XLONG" => {
                    for j in 0..ny {
                        for i in 0..nx {
                            values[j * nx + i] = (lons[i] + 180.0).rem_euclid(360.0) - 180.0;
                        }
                    }
                }
                "COSALPHA" => values.fill(1.0),
                "T" => {
                    if values.iter().any(|v| *v <= 0.0) {
                        return Err("temperature must be positive".into());
                    }
                    if let Some(pressure) = temperature_pressure.as_ref() {
                        for (temperature, pressure) in values.iter_mut().zip(pressure) {
                            *temperature /= (*pressure * 0.00001).powf(WRF_KAPPA);
                        }
                    }
                    values.iter_mut().for_each(|v| *v -= 300.0);
                }
                "P" => {
                    if values.iter().any(|v| *v <= 0.0) {
                        return Err("pressure must be positive".into());
                    }
                }
                "PHB" => {
                    for k in 1..layers {
                        for i in 0..plane {
                            if values[k * plane + i] <= values[(k - 1) * plane + i] {
                                return Err(
                                    "interface height must decrease from native top to surface"
                                        .into(),
                                );
                            }
                        }
                    }
                    values.iter_mut().for_each(|v| *v *= WRF_G);
                }
                "U" => values = faces(&values, nz, ny, nx, true),
                "V" => values = faces(&values, nz, ny, nx, false),
                _ => {}
            }
            if name.starts_with('Q') {
                for (v, q) in values.iter_mut().zip(&qv) {
                    if *v < 0.0 {
                        return Err(format!("native {source} has negative mass or number"));
                    }
                    *v /= 1.0 - q;
                }
            }
            writer
                .write_record(0, id, VarData::F64(&values))
                .map_err(|e| e.to_string())?;
        }
        writer.finish().map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, destination).map_err(|e| e.to_string())?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_axis_flip_preserves_columns() {
        assert_eq!(
            flip(&[1., 2., 3., 4., 5., 6.], 3, 2),
            vec![5., 6., 3., 4., 1., 2.]
        );
    }
    #[test]
    fn face_arrays_keep_edges_and_uniform_wind() {
        assert_eq!(faces(&vec![8.; 24], 2, 3, 4, true), vec![8.; 30]);
        assert_eq!(faces(&vec![-3.; 24], 2, 3, 4, false), vec![-3.; 32]);
    }
}
