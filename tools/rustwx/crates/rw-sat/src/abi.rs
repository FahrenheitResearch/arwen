use chrono::{DateTime, Utc};
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};

use crate::geostationary::{SweepAngleAxis, scan_angles_to_lat_lon};
use crate::goes::{GoesSatellite, parse_goes_abi_filename};
use crate::netcdf::{
    ScaledVariable, open_goes_netcdf_lossy, read_scaled_f32, read_scaled_f32_window,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiSector {
    FullDisk,
    Conus,
    Mesoscale1,
    Mesoscale2,
    Mesoscale,
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbiFixedGrid {
    pub nx: usize,
    pub ny: usize,
    pub x_scan_rad: Vec<f64>,
    pub y_scan_rad: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoesImagerProjection {
    pub perspective_point_height_m: f64,
    pub semi_major_axis_m: f64,
    pub semi_minor_axis_m: f64,
    pub longitude_of_projection_origin_deg: f64,
    pub sweep_angle_axis: SweepAngleAxis,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoesAbiScene {
    pub path: PathBuf,
    pub product: String,
    pub sector: AbiSector,
    pub channel: Option<u8>,
    pub satellite: GoesSatellite,
    pub start_time_utc: DateTime<Utc>,
    pub end_time_utc: DateTime<Utc>,
    pub projection: GoesImagerProjection,
    pub fixed_grid: AbiFixedGrid,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoesAbiField {
    pub scene: GoesAbiScene,
    pub variable_name: String,
    pub units: Option<String>,
    pub values: Vec<f32>,
}

impl GoesAbiScene {
    pub fn lat_lon_mesh(&self) -> (Vec<f32>, Vec<f32>) {
        let len = self.fixed_grid.nx.saturating_mul(self.fixed_grid.ny);
        let mut lat = Vec::with_capacity(len);
        let mut lon = Vec::with_capacity(len);
        for &y in &self.fixed_grid.y_scan_rad {
            for &x in &self.fixed_grid.x_scan_rad {
                match self.projection.scan_angles_to_lat_lon(x, y) {
                    Some((lat_value, lon_value)) => {
                        lat.push(lat_value);
                        lon.push(lon_value);
                    }
                    None => {
                        lat.push(f32::NAN);
                        lon.push(f32::NAN);
                    }
                }
            }
        }
        (lat, lon)
    }

    pub fn approximate_lat_lon_bounds(
        &self,
        max_samples_per_axis: usize,
    ) -> Option<(f64, f64, f64, f64)> {
        let max_samples = max_samples_per_axis.max(2);
        let x_step = (self.fixed_grid.nx / max_samples).max(1);
        let y_step = (self.fixed_grid.ny / max_samples).max(1);
        let mut west = f64::INFINITY;
        let mut east = f64::NEG_INFINITY;
        let mut south = f64::INFINITY;
        let mut north = f64::NEG_INFINITY;
        let mut seen = false;

        let mut rows = (0..self.fixed_grid.ny).step_by(y_step).collect::<Vec<_>>();
        if rows.last().copied() != Some(self.fixed_grid.ny.saturating_sub(1)) {
            rows.push(self.fixed_grid.ny.saturating_sub(1));
        }
        let mut cols = (0..self.fixed_grid.nx).step_by(x_step).collect::<Vec<_>>();
        if cols.last().copied() != Some(self.fixed_grid.nx.saturating_sub(1)) {
            cols.push(self.fixed_grid.nx.saturating_sub(1));
        }

        for row in rows {
            let y = self.fixed_grid.y_scan_rad[row];
            for &col in &cols {
                let x = self.fixed_grid.x_scan_rad[col];
                let Some((lat, lon)) = self.projection.scan_angles_to_lat_lon(x, y) else {
                    continue;
                };
                let lat = f64::from(lat);
                let lon = f64::from(lon);
                if !(lat.is_finite() && lon.is_finite()) {
                    continue;
                }
                west = west.min(lon);
                east = east.max(lon);
                south = south.min(lat);
                north = north.max(lat);
                seen = true;
            }
        }

        seen.then_some((west, east, south, north))
    }
}

impl GoesImagerProjection {
    pub fn scan_angles_to_lat_lon(&self, x_rad: f64, y_rad: f64) -> Option<(f32, f32)> {
        scan_angles_to_lat_lon(
            self.perspective_point_height_m,
            self.semi_major_axis_m,
            self.semi_minor_axis_m,
            self.longitude_of_projection_origin_deg,
            self.sweep_angle_axis,
            x_rad,
            y_rad,
        )
    }
}

pub fn read_goes_abi_scene(path: impl AsRef<Path>) -> Result<GoesAbiScene, Box<dyn Error>> {
    let path = path.as_ref();
    let parsed = parse_goes_abi_filename(path)?;
    let file = open_goes_netcdf_lossy(path)?;
    let x = read_scaled_f32(&file, "x")?;
    let y = read_scaled_f32(&file, "y")?;
    if x.values.is_empty() || y.values.is_empty() {
        return Err(boxed_error(format!(
            "GOES ABI file has empty fixed grid axes: {}",
            path.display()
        )));
    }

    let projection_var = file
        .variable("goes_imager_projection")
        .ok_or_else(|| boxed_error("variable not found: goes_imager_projection"))?;
    let projection = GoesImagerProjection {
        perspective_point_height_m: required_attr_f64(&projection_var, "perspective_point_height")?,
        semi_major_axis_m: required_attr_f64(&projection_var, "semi_major_axis")?,
        semi_minor_axis_m: required_attr_f64(&projection_var, "semi_minor_axis")?,
        longitude_of_projection_origin_deg: required_attr_f64(
            &projection_var,
            "longitude_of_projection_origin",
        )?,
        sweep_angle_axis: projection_var
            .attribute("sweep_angle_axis")
            .and_then(|attr| attr.as_string())
            .map(SweepAngleAxis::parse)
            .unwrap_or(SweepAngleAxis::X),
    };

    Ok(GoesAbiScene {
        path: path.to_path_buf(),
        product: parsed.product.clone(),
        sector: sector_from_product(&parsed.product),
        channel: parsed.channel,
        satellite: parsed.satellite,
        start_time_utc: parsed.start_time_utc,
        end_time_utc: parsed.end_time_utc,
        projection,
        fixed_grid: AbiFixedGrid {
            nx: x.values.len(),
            ny: y.values.len(),
            x_scan_rad: x.values.into_iter().map(f64::from).collect(),
            y_scan_rad: y.values.into_iter().map(f64::from).collect(),
        },
    })
}

pub fn read_goes_abi_field(
    path: impl AsRef<Path>,
    variable_name: &str,
) -> Result<GoesAbiField, Box<dyn Error>> {
    let path = path.as_ref();
    let scene = read_goes_abi_scene(path)?;
    let file = open_goes_netcdf_lossy(path)?;
    let variable = read_scaled_f32(&file, variable_name)?;
    validate_field_shape(&scene, &variable)?;
    Ok(GoesAbiField {
        scene,
        variable_name: variable_name.to_string(),
        units: variable.units,
        values: variable.values,
    })
}

pub fn read_goes_abi_field_window(
    path: impl AsRef<Path>,
    variable_name: &str,
    x_start: usize,
    x_count: usize,
    y_start: usize,
    y_count: usize,
) -> Result<GoesAbiField, Box<dyn Error>> {
    if x_count == 0 || y_count == 0 {
        return Err(boxed_error(format!(
            "empty GOES ABI window requested for {variable_name}: x_count={x_count} y_count={y_count}"
        )));
    }
    let path = path.as_ref();
    let mut scene = read_goes_abi_scene(path)?;
    if x_start.saturating_add(x_count) > scene.fixed_grid.nx
        || y_start.saturating_add(y_count) > scene.fixed_grid.ny
    {
        return Err(boxed_error(format!(
            "GOES ABI window {x_start}..{} x {y_start}..{} exceeds grid {}x{}",
            x_start.saturating_add(x_count),
            y_start.saturating_add(y_count),
            scene.fixed_grid.nx,
            scene.fixed_grid.ny
        )));
    }
    let file = open_goes_netcdf_lossy(path)?;
    let variable =
        read_scaled_f32_window(&file, variable_name, y_start, y_count, x_start, x_count)?;
    validate_window_shape(variable_name, &variable, x_count, y_count)?;
    scene.fixed_grid = AbiFixedGrid {
        nx: x_count,
        ny: y_count,
        x_scan_rad: scene.fixed_grid.x_scan_rad[x_start..x_start + x_count].to_vec(),
        y_scan_rad: scene.fixed_grid.y_scan_rad[y_start..y_start + y_count].to_vec(),
    };
    Ok(GoesAbiField {
        scene,
        variable_name: variable_name.to_string(),
        units: variable.units,
        values: variable.values,
    })
}

fn validate_field_shape(
    scene: &GoesAbiScene,
    variable: &ScaledVariable,
) -> Result<(), Box<dyn Error>> {
    let expected_len = scene.fixed_grid.nx.saturating_mul(scene.fixed_grid.ny);
    if variable.values.len() != expected_len {
        return Err(boxed_error(format!(
            "GOES ABI variable {} length {} does not match grid {}x{}",
            variable.name,
            variable.values.len(),
            scene.fixed_grid.nx,
            scene.fixed_grid.ny
        )));
    }
    let shape_matches = match variable.shape.as_slice() {
        [ny, nx] => *nx == scene.fixed_grid.nx && *ny == scene.fixed_grid.ny,
        [len] => *len == expected_len,
        _ => false,
    };
    if !shape_matches {
        return Err(boxed_error(format!(
            "GOES ABI variable {} shape {:?} does not match grid {}x{}",
            variable.name, variable.shape, scene.fixed_grid.nx, scene.fixed_grid.ny
        )));
    }
    Ok(())
}

fn validate_window_shape(
    variable_name: &str,
    variable: &ScaledVariable,
    x_count: usize,
    y_count: usize,
) -> Result<(), Box<dyn Error>> {
    let expected_len = x_count.saturating_mul(y_count);
    if variable.values.len() != expected_len {
        return Err(boxed_error(format!(
            "GOES ABI variable {variable_name} window length {} does not match grid {}x{}",
            variable.values.len(),
            x_count,
            y_count
        )));
    }
    let shape_matches = match variable.shape.as_slice() {
        [ny, nx] => *nx == x_count && *ny == y_count,
        [len] => *len == expected_len,
        _ => false,
    };
    if !shape_matches {
        return Err(boxed_error(format!(
            "GOES ABI variable {variable_name} window shape {:?} does not match grid {}x{}",
            variable.shape, x_count, y_count
        )));
    }
    Ok(())
}

fn required_attr_f64(variable: &netcrust::Variable, name: &str) -> Result<f64, Box<dyn Error>> {
    variable
        .attribute(name)
        .and_then(|attr| attr.as_f64())
        .ok_or_else(|| boxed_error(format!("missing numeric projection attribute: {name}")))
}

fn sector_from_product(product: &str) -> AbiSector {
    let upper = product.to_ascii_uppercase();
    if upper.ends_with("M1") {
        AbiSector::Mesoscale1
    } else if upper.ends_with("M2") {
        AbiSector::Mesoscale2
    } else if upper.ends_with('M') {
        AbiSector::Mesoscale
    } else if upper.ends_with('C') {
        AbiSector::Conus
    } else if upper.ends_with('F') {
        AbiSector::FullDisk
    } else {
        AbiSector::Unknown(product.to_string())
    }
}

fn boxed_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_sector_from_product_name() {
        assert_eq!(sector_from_product("ABI-L2-MCMIPC"), AbiSector::Conus);
        assert_eq!(sector_from_product("ABI-L2-CMIPF"), AbiSector::FullDisk);
        assert_eq!(sector_from_product("ABI-L2-CMIPM1"), AbiSector::Mesoscale1);
        assert_eq!(sector_from_product("ABI-L2-CMIPM2"), AbiSector::Mesoscale2);
        assert_eq!(sector_from_product("ABI-L2-CMIPM"), AbiSector::Mesoscale);
    }

    #[test]
    #[ignore]
    fn reads_real_goes_abi_fixture() {
        let path = std::env::var_os("RUSTWX_GOES_ABI_FIXTURE")
            .expect("set RUSTWX_GOES_ABI_FIXTURE to a GOES ABI NetCDF file");
        let scene = read_goes_abi_scene(PathBuf::from(path)).unwrap();
        assert_eq!(scene.fixed_grid.nx, 2500);
        assert_eq!(scene.fixed_grid.ny, 1500);
        let center_x = scene.fixed_grid.x_scan_rad[scene.fixed_grid.nx / 2];
        let center_y = scene.fixed_grid.y_scan_rad[scene.fixed_grid.ny / 2];
        let (lat, lon) = scene
            .projection
            .scan_angles_to_lat_lon(center_x, center_y)
            .expect("center point should intersect earth");
        assert!(lat.is_finite());
        assert!(lon.is_finite());
        let field = read_goes_abi_field(scene.path.clone(), "CMI").unwrap();
        assert_eq!(field.values.len(), 2500 * 1500);
    }
}
