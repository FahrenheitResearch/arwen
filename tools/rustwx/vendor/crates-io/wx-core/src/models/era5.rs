/// Configuration and URL generation for ERA5 (ECMWF Reanalysis v5).
///
/// ERA5 is the gold standard for historical weather reanalysis.
/// 0.25° global grid, hourly data from 1940 to near-present.
///
/// **Note:** ERA5 data on AWS is stored as NetCDF (not GRIB2).
/// The CDS API provides GRIB format but requires an API key.
/// Google Cloud also mirrors ERA5 in Zarr format.
pub struct Era5Config;

impl Era5Config {
    /// AWS Open Data URL for ERA5 NetCDF files (single-level variables).
    ///
    /// - `year`: 4-digit year (e.g. `"2024"`)
    /// - `month`: 2-digit month (e.g. `"03"`)
    /// - `variable`: ERA5 variable name (e.g. `"2m_temperature"`, `"mean_sea_level_pressure"`)
    ///
    /// Returns a NetCDF URL — **not GRIB2**. Each file contains one month of hourly data.
    pub fn aws_url(year: &str, month: &str, variable: &str) -> String {
        format!(
            "https://era5-pds.s3.amazonaws.com/{}/{}/data/{}.nc",
            year, month, variable
        )
    }

    /// Google Cloud Storage URL for ERA5 in ARCO (Analysis-Ready, Cloud-Optimized) Zarr format.
    ///
    /// - `variable`: ERA5 variable name
    /// - `level_type`: `"single-level"` or `"pressure-level"`
    pub fn gcs_zarr_url(level_type: &str) -> String {
        match level_type {
            "pressure-level" | "pressure" | "prs" => {
                "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3".to_string()
            }
            _ => {
                "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3".to_string()
            }
        }
    }

    /// CDS API endpoint for retrieving ERA5 GRIB data (requires API key).
    pub fn cds_api_url() -> &'static str {
        "https://cds.climate.copernicus.eu/api/retrieve/reanalysis-era5-single-levels"
    }

    /// ERA5 does not have traditional GRIB2 .idx files on AWS.
    /// Returns `None` since the AWS mirror uses NetCDF format.
    pub fn idx_url(_year: &str, _month: &str, _variable: &str) -> Option<String> {
        None
    }

    // --- Grid specifications (0.25 degree global) ---

    pub fn grid_nx() -> u32 { 1440 }
    pub fn grid_ny() -> u32 { 721 }
    pub fn grid_dx() -> f64 { 0.25 } // degrees
    pub fn grid_dy() -> f64 { 0.25 }

    // --- Common variable names (ERA5 naming convention for AWS NetCDF files) ---

    pub fn var_temperature_2m() -> &'static str { "2m_temperature" }
    pub fn var_dewpoint_2m() -> &'static str { "2m_dewpoint_temperature" }
    pub fn var_u_wind_10m() -> &'static str { "10m_u_component_of_wind" }
    pub fn var_v_wind_10m() -> &'static str { "10m_v_component_of_wind" }
    pub fn var_mslp() -> &'static str { "mean_sea_level_pressure" }
    pub fn var_surface_pressure() -> &'static str { "surface_pressure" }
    pub fn var_total_precip() -> &'static str { "total_precipitation" }
    pub fn var_total_cloud_cover() -> &'static str { "total_cloud_cover" }
    pub fn var_ssrd() -> &'static str { "surface_solar_radiation_downwards" }
    pub fn var_ssr() -> &'static str { "surface_net_solar_radiation" }
    pub fn var_cape() -> &'static str { "convective_available_potential_energy" }
    pub fn var_boundary_layer_height() -> &'static str { "boundary_layer_height" }
    pub fn var_snow_depth() -> &'static str { "snow_depth" }
    pub fn var_soil_temp_level1() -> &'static str { "soil_temperature_level_1" }
    pub fn var_sea_surface_temp() -> &'static str { "sea_surface_temperature" }

    /// Common pressure-level variable short names (for CDS requests).
    pub fn prs_var(short_name: &str, level_hpa: u32) -> String {
        format!("{}:{}", short_name, level_hpa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_url_format() {
        let url = Era5Config::aws_url("2024", "03", "2m_temperature");
        assert_eq!(
            url,
            "https://era5-pds.s3.amazonaws.com/2024/03/data/2m_temperature.nc"
        );
    }

    #[test]
    fn test_gcs_zarr_url() {
        let url = Era5Config::gcs_zarr_url("single-level");
        assert!(url.starts_with("gs://gcp-public-data-arco-era5/"));
    }

    #[test]
    fn test_gcs_zarr_url_pressure() {
        let url = Era5Config::gcs_zarr_url("pressure-level");
        assert!(url.contains("zarr"));
    }

    #[test]
    fn test_cds_api_url() {
        let url = Era5Config::cds_api_url();
        assert!(url.contains("cds.climate.copernicus.eu"));
    }

    #[test]
    fn test_idx_url_returns_none() {
        assert!(Era5Config::idx_url("2024", "03", "2m_temperature").is_none());
    }

    #[test]
    fn test_grid_specs() {
        assert_eq!(Era5Config::grid_nx(), 1440);
        assert_eq!(Era5Config::grid_ny(), 721);
        assert_eq!(Era5Config::grid_dx(), 0.25);
    }

    #[test]
    fn test_variable_names() {
        assert_eq!(Era5Config::var_temperature_2m(), "2m_temperature");
        assert_eq!(Era5Config::var_mslp(), "mean_sea_level_pressure");
    }

    #[test]
    fn test_prs_var() {
        assert_eq!(Era5Config::prs_var("t", 500), "t:500");
    }
}
