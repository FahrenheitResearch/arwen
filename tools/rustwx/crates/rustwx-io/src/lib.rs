mod cache;

pub use cache::{
    CachedFetchMetadata, CachedFetchResult, CachedFieldResult, artifact_cache_dir,
    fetch_cache_paths, field_cache_path, load_cached_fetch, load_cached_raw_fetch,
    load_cached_selected_field, raw_fetch_cache_paths, store_cached_fetch, store_cached_raw_fetch,
    store_cached_selected_field,
};

use grib_core::grib2::{
    Grib2File, Grib2Message, GridDefinition, flip_rows, grid_latlon, unpack_message,
};
use hdf5_reader::{Datatype, Hdf5File};
use rayon::prelude::*;
use rustwx_core::{
    CanonicalField, FieldProduct, FieldSelector, GridProjection, GridShape, LatLonGrid, ModelId,
    ModelRunRequest, ModelTimestep, ProbabilitySelection, ResolvedUrl, SelectedField2D,
    SelectedHybridLevelVolume, SourceId, VerticalSelector,
};
use rustwx_models::{latest_available_run, model_summary, resolve_urls};
use serde::Serialize;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;
use wx_core::download::{DownloadClient, byte_ranges, find_entries, parse_idx};

const FETCH_CACHE_LOCK_STALE_AFTER: Duration = Duration::from_secs(30 * 60);
const FETCH_CACHE_LOCK_WAIT_TIMEOUT: Duration = Duration::from_secs(45 * 60);
const FETCH_CACHE_LOCK_RETRY_AFTER: Duration = Duration::from_millis(250);

#[derive(Debug, Error)]
pub enum IoError {
    #[error(transparent)]
    Core(#[from] rustwx_core::RustwxError),
    #[error(transparent)]
    Model(#[from] rustwx_models::ModelError),
    #[error("download client error: {0}")]
    Download(String),
    #[error("cache error: {0}")]
    Cache(String),
    #[error("grib error: {0}")]
    Grib(String),
    #[error("field '{selector}' was not found in GRIB data")]
    FieldNotFound { selector: FieldSelector },
    #[error("selector '{selector}' is not supported by structured GRIB extraction")]
    UnsupportedStructuredSelector { selector: FieldSelector },
    #[error("grid coordinates could not be derived for selector '{selector}'")]
    MissingGridCoordinates { selector: FieldSelector },
    #[error("wrf error: {0}")]
    Wrf(String),
    #[error("ODIM HDF5 error: {0}")]
    Odim(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProbeResult {
    pub source: SourceId,
    pub available: bool,
    pub grib_url: String,
    pub idx_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct FetchRequest {
    pub request: ModelRunRequest,
    pub source_override: Option<SourceId>,
    pub variable_patterns: Vec<String>,
}

impl FetchRequest {
    pub fn from_timestep<S, I, P>(
        timestep: &ModelTimestep,
        product: S,
        source_override: Option<SourceId>,
        variable_patterns: I,
    ) -> Result<Self, rustwx_core::RustwxError>
    where
        S: Into<String>,
        I: IntoIterator<Item = P>,
        P: Into<String>,
    {
        Ok(Self {
            request: timestep.request(product)?,
            source_override,
            variable_patterns: variable_patterns.into_iter().map(Into::into).collect(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct FetchResult {
    pub source: SourceId,
    pub url: String,
    pub bytes: Vec<u8>,
}

pub fn grid_projection_from_grib2_grid(grid: &GridDefinition) -> Option<GridProjection> {
    match grid.template {
        0 | 1 | 40 => Some(GridProjection::Geographic),
        10 => Some(GridProjection::Mercator {
            latitude_of_true_scale_deg: grid.latin1,
            central_meridian_deg: normalize_longitude(longitude_midpoint(grid.lon1, grid.lon2)),
        }),
        20 => Some(GridProjection::PolarStereographic {
            true_latitude_deg: if grid.lad != 0.0 {
                grid.lad
            } else {
                grid.latin1
            },
            central_meridian_deg: normalize_longitude(grid.lov),
            south_pole_on_projection_plane: (grid.projection_center_flag & 1) != 0,
        }),
        30 => Some(GridProjection::LambertConformal {
            standard_parallel_1_deg: grid.latin1,
            standard_parallel_2_deg: if grid.latin2 != 0.0 {
                grid.latin2
            } else {
                grid.latin1
            },
            central_meridian_deg: normalize_longitude(grid.lov),
        }),
        template => Some(GridProjection::Other { template }),
    }
}

pub fn client() -> Result<DownloadClient, IoError> {
    // rustwx owns fetch/decode caching through the explicit cache_root passed
    // into fetch_bytes_with_cache. Enabling wx-core's default cache here writes
    // duplicate GRIB bytes to platform locations such as ~/.cache/metrust, which
    // bypasses callers' storage controls on research nodes.
    DownloadClient::new().map_err(|err| IoError::Download(err.to_string()))
}

pub fn latest_run(
    model: ModelId,
    date_yyyymmdd: &str,
) -> Result<rustwx_models::LatestRun, IoError> {
    latest_available_run(model, None, date_yyyymmdd).map_err(Into::into)
}

pub fn probe_sources(fetch: &FetchRequest) -> Result<Vec<ProbeResult>, IoError> {
    let client = client()?;
    let urls = filtered_urls(fetch)?;
    Ok(urls
        .into_iter()
        .map(|resolved| {
            let available = probe_availability(&client, &resolved);
            ProbeResult {
                source: resolved.source,
                available,
                grib_url: resolved.grib_url,
                idx_url: resolved.idx_url,
            }
        })
        .collect())
}

pub fn available_forecast_hours(
    model: ModelId,
    date_yyyymmdd: &str,
    hour_utc: u8,
    product: &str,
    source_override: Option<SourceId>,
) -> Result<Vec<u16>, IoError> {
    let candidates = candidate_hours(model, hour_utc);
    available_forecast_hours_for_candidates(
        model,
        date_yyyymmdd,
        hour_utc,
        product,
        source_override,
        &candidates,
    )
}

pub fn available_forecast_hours_for_candidates(
    model: ModelId,
    date_yyyymmdd: &str,
    hour_utc: u8,
    product: &str,
    source_override: Option<SourceId>,
    candidates: &[u16],
) -> Result<Vec<u16>, IoError> {
    let client = client()?;
    let summary = model_summary(model);

    let parallelize = candidates.len() <= 48
        || should_parallelize_hour_availability_probes(source_override, summary);
    let available = if parallelize {
        candidates
            .par_iter()
            .filter_map(|&forecast_hour| {
                let cycle = rustwx_core::CycleSpec::new(date_yyyymmdd, hour_utc).ok()?;
                let fetch = FetchRequest {
                    request: ModelRunRequest::new(model, cycle, forecast_hour, product).ok()?,
                    source_override,
                    variable_patterns: Vec::new(),
                };
                if fetch_request_is_available(&client, &fetch).ok()? {
                    Some(forecast_hour)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    } else {
        candidates
            .iter()
            .filter_map(|&forecast_hour| {
                let cycle = rustwx_core::CycleSpec::new(date_yyyymmdd, hour_utc).ok()?;
                let fetch = FetchRequest {
                    request: ModelRunRequest::new(model, cycle, forecast_hour, product).ok()?,
                    source_override,
                    variable_patterns: Vec::new(),
                };
                if fetch_request_is_available(&client, &fetch).ok()? {
                    Some(forecast_hour)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };

    let mut available = available;
    available.sort_unstable();
    Ok(available)
}

pub fn mrms_latest_product_url(product: &str) -> Result<String, IoError> {
    validate_mrms_product_token(product)?;
    Ok(format!(
        "https://mrms.ncep.noaa.gov/2D/{product}/MRMS_{product}.latest.grib2.gz"
    ))
}

pub fn fetch_mrms_latest_product(product: &str) -> Result<Vec<u8>, IoError> {
    let url = mrms_latest_product_url(product)?;
    let bytes = client()?
        .get_bytes(&url)
        .map_err(|err| IoError::Download(err.to_string()))?;
    maybe_decompress_grib_payload(&url, bytes).map_err(IoError::Download)
}

pub fn extract_mrms_latest_reflectivity_at_lowest_altitude() -> Result<SelectedField2D, IoError> {
    let bytes = fetch_mrms_latest_product("ReflectivityAtLowestAltitude")?;
    extract_field_from_bytes(
        &bytes,
        FieldSelector::altitude_msl(CanonicalField::RadarReflectivity, 500),
    )
}

pub fn extract_mrms_latest_composite_reflectivity() -> Result<SelectedField2D, IoError> {
    let bytes = fetch_mrms_latest_product("MergedReflectivityQCComposite")?;
    extract_field_from_bytes(
        &bytes,
        FieldSelector::altitude_msl(CanonicalField::CompositeReflectivity, 500),
    )
}

fn validate_mrms_product_token(product: &str) -> Result<(), IoError> {
    let valid = !product.is_empty()
        && product
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    if valid {
        Ok(())
    } else {
        Err(IoError::Download(format!(
            "invalid MRMS product token '{product}'"
        )))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OperaDownloadLink {
    pub href: String,
    pub title: Option<String>,
    pub length: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OperaRadarMeta {
    pub projdef: String,
    pub xsize: usize,
    pub ysize: usize,
    pub xscale_m: f64,
    pub yscale_m: f64,
    pub ll_lon_deg: f64,
    pub ll_lat_deg: f64,
    pub ul_lon_deg: f64,
    pub ul_lat_deg: f64,
    pub ur_lon_deg: f64,
    pub ur_lat_deg: f64,
    pub lr_lon_deg: f64,
    pub lr_lat_deg: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OperaDbzhCoverage {
    pub download_links: Vec<OperaDownloadLink>,
    pub radar_meta: Option<OperaRadarMeta>,
}

impl OperaDbzhCoverage {
    pub fn latest_odim_link(&self) -> Option<&OperaDownloadLink> {
        self.download_links.last()
    }
}

pub fn eumetnet_opera_dbzh_coverage_url(datetime_range: &str) -> Result<String, IoError> {
    validate_eumetnet_datetime_range(datetime_range)?;
    Ok(format!(
        "https://api.meteogate.eu/eu-eumetnet-weather-radar/collections/observations/locations/0-20010-0-OPERA?datetime={}&f=CoverageJSON&standard_name=DBZH&format=ODIM&method=comp",
        encode_query_component(datetime_range)
    ))
}

pub fn fetch_eumetnet_opera_dbzh_coverage(
    datetime_range: &str,
) -> Result<OperaDbzhCoverage, IoError> {
    let url = eumetnet_opera_dbzh_coverage_url(datetime_range)?;
    let bytes = client()?
        .get_bytes(&url)
        .map_err(|err| IoError::Download(err.to_string()))?;
    parse_eumetnet_opera_dbzh_coverage_json(&bytes)
}

pub fn fetch_eumetnet_opera_latest_dbzh_for_range(
    datetime_range: &str,
) -> Result<SelectedField2D, IoError> {
    let coverage = fetch_eumetnet_opera_dbzh_coverage(datetime_range)?;
    let link = coverage.latest_odim_link().ok_or_else(|| {
        IoError::Download("EUMETNET OPERA coverage has no ODIM HDF5 links".into())
    })?;
    let bytes = fetch_eumetnet_opera_odim_h5(&link.href)?;
    extract_eumetnet_opera_dbzh_from_odim_h5(&bytes)
}

pub fn fetch_eumetnet_opera_odim_h5(url: &str) -> Result<Vec<u8>, IoError> {
    validate_eumetnet_opera_odim_url(url)?;
    client()?
        .get_bytes(url)
        .map_err(|err| IoError::Download(err.to_string()))
}

pub fn extract_eumetnet_opera_dbzh_from_odim_h5(bytes: &[u8]) -> Result<SelectedField2D, IoError> {
    let file = Hdf5File::from_bytes(bytes).map_err(|err| IoError::Odim(err.to_string()))?;
    let dataset = file
        .dataset("/dataset1/data1/data")
        .map_err(|err| IoError::Odim(format!("missing /dataset1/data1/data: {err}")))?;
    let shape = dataset.shape();
    let &[ny_u64, nx_u64] = shape else {
        return Err(IoError::Odim(format!(
            "OPERA DBZH dataset must be 2D, got shape {shape:?}"
        )));
    };
    let nx = usize::try_from(nx_u64)
        .map_err(|_| IoError::Odim(format!("OPERA x size exceeds usize: {nx_u64}")))?;
    let ny = usize::try_from(ny_u64)
        .map_err(|_| IoError::Odim(format!("OPERA y size exceeds usize: {ny_u64}")))?;

    let data_what = file
        .group("/dataset1/data1/what")
        .map_err(|err| IoError::Odim(format!("missing /dataset1/data1/what: {err}")))?;
    let quantity = hdf5_group_attr_string(&data_what, "quantity")?;
    if quantity != "DBZH" {
        return Err(IoError::Odim(format!(
            "expected OPERA quantity DBZH, got {quantity}"
        )));
    }
    let gain = hdf5_group_attr_f64_default(&data_what, "gain", 1.0)?;
    let offset = hdf5_group_attr_f64_default(&data_what, "offset", 0.0)?;
    let nodata = hdf5_group_attr_f64_optional(&data_what, "nodata")?;
    let undetect = hdf5_group_attr_f64_optional(&data_what, "undetect")?;

    let raw = hdf5_dataset_values_f64(&dataset)?;
    let values = raw
        .into_iter()
        .map(|value| {
            let missing = !value.is_finite()
                || nodata.is_some_and(|sentinel| (value - sentinel).abs() < 0.5)
                || undetect.is_some_and(|sentinel| (value - sentinel).abs() < 0.5);
            if missing {
                f32::NAN
            } else {
                (value * gain + offset) as f32
            }
        })
        .collect::<Vec<_>>();

    let meta = opera_radar_meta_from_hdf5(&file)?;
    if meta.xsize != nx || meta.ysize != ny {
        return Err(IoError::Odim(format!(
            "OPERA data shape {nx}x{ny} does not match metadata {}x{}",
            meta.xsize, meta.ysize
        )));
    }
    let grid = opera_laea_latlon_grid(&meta)?;
    SelectedField2D::new(
        FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity),
        "dBZ",
        grid,
        values,
    )
    .map_err(Into::into)
}

fn parse_eumetnet_opera_dbzh_coverage_json(bytes: &[u8]) -> Result<OperaDbzhCoverage, IoError> {
    let root: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|err| IoError::Odim(format!("invalid EUMETNET CoverageJSON: {err}")))?;
    let download_links = root
        .get("links")
        .and_then(serde_json::Value::as_array)
        .map(|links| {
            links
                .iter()
                .filter_map(|link| {
                    let href = link.get("href")?.as_str()?;
                    let mime = link.get("type").and_then(serde_json::Value::as_str);
                    if mime != Some("application/x-odim") && !href.ends_with(".h5") {
                        return None;
                    }
                    Some(OperaDownloadLink {
                        href: href.to_string(),
                        title: link
                            .get("title")
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_string),
                        length: link.get("length").and_then(serde_json::Value::as_u64),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let radar_meta = root
        .get("metocean:radar_meta")
        .map(opera_radar_meta_from_json)
        .transpose()?;

    Ok(OperaDbzhCoverage {
        download_links,
        radar_meta,
    })
}

fn opera_radar_meta_from_json(value: &serde_json::Value) -> Result<OperaRadarMeta, IoError> {
    Ok(OperaRadarMeta {
        projdef: json_string(value, "projdef")?,
        xsize: json_usize(value, "xsize")?,
        ysize: json_usize(value, "ysize")?,
        xscale_m: json_f64(value, "xscale")?,
        yscale_m: json_f64(value, "yscale")?,
        ll_lon_deg: json_f64(value, "LL_lon")?,
        ll_lat_deg: json_f64(value, "LL_lat")?,
        ul_lon_deg: json_f64(value, "UL_lon")?,
        ul_lat_deg: json_f64(value, "UL_lat")?,
        ur_lon_deg: json_f64(value, "UR_lon")?,
        ur_lat_deg: json_f64(value, "UR_lat")?,
        lr_lon_deg: json_f64(value, "LR_lon")?,
        lr_lat_deg: json_f64(value, "LR_lat")?,
    })
}

fn opera_radar_meta_from_hdf5(file: &Hdf5File) -> Result<OperaRadarMeta, IoError> {
    let where_group = file
        .group("/where")
        .map_err(|err| IoError::Odim(format!("missing /where group: {err}")))?;
    Ok(OperaRadarMeta {
        projdef: hdf5_group_attr_string(&where_group, "projdef")?,
        xsize: hdf5_group_attr_usize(&where_group, "xsize")?,
        ysize: hdf5_group_attr_usize(&where_group, "ysize")?,
        xscale_m: hdf5_group_attr_f64(&where_group, "xscale")?,
        yscale_m: hdf5_group_attr_f64(&where_group, "yscale")?,
        ll_lon_deg: hdf5_group_attr_f64(&where_group, "LL_lon")?,
        ll_lat_deg: hdf5_group_attr_f64(&where_group, "LL_lat")?,
        ul_lon_deg: hdf5_group_attr_f64(&where_group, "UL_lon")?,
        ul_lat_deg: hdf5_group_attr_f64(&where_group, "UL_lat")?,
        ur_lon_deg: hdf5_group_attr_f64(&where_group, "UR_lon")?,
        ur_lat_deg: hdf5_group_attr_f64(&where_group, "UR_lat")?,
        lr_lon_deg: hdf5_group_attr_f64(&where_group, "LR_lon")?,
        lr_lat_deg: hdf5_group_attr_f64(&where_group, "LR_lat")?,
    })
}

fn opera_laea_latlon_grid(meta: &OperaRadarMeta) -> Result<LatLonGrid, IoError> {
    if !meta
        .projdef
        .split_whitespace()
        .any(|part| part == "+proj=laea")
    {
        return Err(IoError::Odim(format!(
            "OPERA projection is not LAEA: {}",
            meta.projdef
        )));
    }
    let lat0 = projdef_value(&meta.projdef, "+lat_0=")
        .ok_or_else(|| IoError::Odim(format!("missing +lat_0 in {}", meta.projdef)))?
        .to_radians();
    let lon0 = projdef_value(&meta.projdef, "+lon_0=")
        .ok_or_else(|| IoError::Odim(format!("missing +lon_0 in {}", meta.projdef)))?
        .to_radians();
    let false_easting = projdef_value(&meta.projdef, "+x_0=")
        .ok_or_else(|| IoError::Odim(format!("missing +x_0 in {}", meta.projdef)))?;
    let false_northing = projdef_value(&meta.projdef, "+y_0=")
        .ok_or_else(|| IoError::Odim(format!("missing +y_0 in {}", meta.projdef)))?;
    let shape = GridShape::new(meta.xsize, meta.ysize)?;
    let mut lat_deg = Vec::with_capacity(shape.len());
    let mut lon_deg = Vec::with_capacity(shape.len());
    for y in 0..meta.ysize {
        let projected_y = -((y as f64) + 0.5) * meta.yscale_m;
        for x in 0..meta.xsize {
            let projected_x = ((x as f64) + 0.5) * meta.xscale_m;
            let (lat, lon) = inverse_spherical_laea(
                projected_x,
                projected_y,
                lat0,
                lon0,
                false_easting,
                false_northing,
            );
            lat_deg.push(lat as f32);
            lon_deg.push(lon as f32);
        }
    }
    LatLonGrid::new(shape, lat_deg, lon_deg).map_err(Into::into)
}

fn inverse_spherical_laea(
    x: f64,
    y: f64,
    lat0: f64,
    lon0: f64,
    false_easting: f64,
    false_northing: f64,
) -> (f64, f64) {
    const AUTHALIC_RADIUS_M: f64 = 6_371_007.181;
    let dx = x - false_easting;
    let dy = y - false_northing;
    let rho = dx.hypot(dy);
    if rho <= f64::EPSILON {
        return (lat0.to_degrees(), normalize_longitude(lon0.to_degrees()));
    }
    let c = 2.0 * (rho / (2.0 * AUTHALIC_RADIUS_M)).clamp(-1.0, 1.0).asin();
    let sin_c = c.sin();
    let cos_c = c.cos();
    let sin_lat0 = lat0.sin();
    let cos_lat0 = lat0.cos();
    let lat = (cos_c * sin_lat0 + dy * sin_c * cos_lat0 / rho).asin();
    let lon = lon0 + (dx * sin_c).atan2(rho * cos_lat0 * cos_c - dy * sin_lat0 * sin_c);
    (lat.to_degrees(), normalize_longitude(lon.to_degrees()))
}

fn projdef_value(projdef: &str, key: &str) -> Option<f64> {
    projdef
        .split_whitespace()
        .find_map(|part| part.strip_prefix(key)?.parse::<f64>().ok())
}

fn hdf5_dataset_values_f64(dataset: &hdf5_reader::Dataset) -> Result<Vec<f64>, IoError> {
    match dataset.dtype() {
        Datatype::FloatingPoint { size: 4, .. } => Ok(dataset
            .read_array::<f32>()
            .map_err(|err| IoError::Odim(err.to_string()))?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FloatingPoint { size: 8, .. } => Ok(dataset
            .read_array::<f64>()
            .map_err(|err| IoError::Odim(err.to_string()))?
            .iter()
            .copied()
            .collect()),
        Datatype::FixedPoint {
            size: 1,
            signed: true,
            ..
        } => Ok(dataset
            .read_array::<i8>()
            .map_err(|err| IoError::Odim(err.to_string()))?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 1,
            signed: false,
            ..
        } => Ok(dataset
            .read_array::<u8>()
            .map_err(|err| IoError::Odim(err.to_string()))?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 2,
            signed: true,
            ..
        } => Ok(dataset
            .read_array::<i16>()
            .map_err(|err| IoError::Odim(err.to_string()))?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 2,
            signed: false,
            ..
        } => Ok(dataset
            .read_array::<u16>()
            .map_err(|err| IoError::Odim(err.to_string()))?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 4,
            signed: true,
            ..
        } => Ok(dataset
            .read_array::<i32>()
            .map_err(|err| IoError::Odim(err.to_string()))?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        Datatype::FixedPoint {
            size: 4,
            signed: false,
            ..
        } => Ok(dataset
            .read_array::<u32>()
            .map_err(|err| IoError::Odim(err.to_string()))?
            .iter()
            .map(|&value| f64::from(value))
            .collect()),
        dtype => Err(IoError::Odim(format!(
            "unsupported OPERA DBZH dataset dtype: {dtype:?}"
        ))),
    }
}

fn hdf5_group_attr_f64(group: &hdf5_reader::group::Group, name: &str) -> Result<f64, IoError> {
    group
        .attribute(name)
        .map_err(|err| IoError::Odim(format!("missing attribute {}@{name}: {err}", group.name())))?
        .read_as_f64()
        .map_err(|err| IoError::Odim(format!("invalid numeric attribute {name}: {err}")))
}

fn hdf5_group_attr_f64_optional(
    group: &hdf5_reader::group::Group,
    name: &str,
) -> Result<Option<f64>, IoError> {
    match group.attribute(name) {
        Ok(attr) => attr
            .read_as_f64()
            .map(Some)
            .map_err(|err| IoError::Odim(format!("invalid numeric attribute {name}: {err}"))),
        Err(_) => Ok(None),
    }
}

fn hdf5_group_attr_f64_default(
    group: &hdf5_reader::group::Group,
    name: &str,
    default: f64,
) -> Result<f64, IoError> {
    Ok(hdf5_group_attr_f64_optional(group, name)?.unwrap_or(default))
}

fn hdf5_group_attr_usize(group: &hdf5_reader::group::Group, name: &str) -> Result<usize, IoError> {
    let value = hdf5_group_attr_f64(group, name)?;
    if value < 0.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(IoError::Odim(format!(
            "attribute {name} cannot be represented as usize: {value}"
        )));
    }
    Ok(value as usize)
}

fn hdf5_group_attr_string(
    group: &hdf5_reader::group::Group,
    name: &str,
) -> Result<String, IoError> {
    group
        .attribute(name)
        .map_err(|err| IoError::Odim(format!("missing attribute {}@{name}: {err}", group.name())))?
        .read_string()
        .map_err(|err| IoError::Odim(format!("invalid string attribute {name}: {err}")))
}

fn validate_eumetnet_datetime_range(datetime_range: &str) -> Result<(), IoError> {
    let valid = !datetime_range.is_empty()
        && datetime_range
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b':' | b'/' | b'.'));
    if valid {
        Ok(())
    } else {
        Err(IoError::Download(format!(
            "invalid EUMETNET datetime range '{datetime_range}'"
        )))
    }
}

fn validate_eumetnet_opera_odim_url(url: &str) -> Result<(), IoError> {
    if url.starts_with("https://s3.waw3-1.cloudferro.com/openradar-24h/") && url.ends_with(".h5") {
        Ok(())
    } else {
        Err(IoError::Download(format!(
            "invalid EUMETNET OPERA ODIM URL '{url}'"
        )))
    }
}

fn encode_query_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn json_string(value: &serde_json::Value, key: &str) -> Result<String, IoError> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| IoError::Odim(format!("missing string JSON field {key}")))
}

fn json_f64(value: &serde_json::Value, key: &str) -> Result<f64, IoError> {
    value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| IoError::Odim(format!("missing numeric JSON field {key}")))
}

fn json_usize(value: &serde_json::Value, key: &str) -> Result<usize, IoError> {
    let raw = value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| IoError::Odim(format!("missing unsigned JSON field {key}")))?;
    usize::try_from(raw).map_err(|_| IoError::Odim(format!("{key} exceeds usize: {raw}")))
}

pub fn fetch_bytes(fetch: &FetchRequest) -> Result<FetchResult, IoError> {
    let client = client()?;
    let urls = filtered_urls(fetch)?;
    let patterns = fetch
        .variable_patterns
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();

    let mut errors = Vec::new();
    for resolved in urls {
        match try_fetch_one(&client, &resolved, &patterns) {
            Ok(bytes) => {
                return Ok(FetchResult {
                    source: resolved.source,
                    url: resolved.grib_url,
                    bytes,
                });
            }
            Err(err) => errors.push(format!("{}: {}", resolved.source, err)),
        }
    }

    Err(IoError::Download(format!(
        "all sources failed for {} f{:03}: {}",
        fetch.request.model,
        fetch.request.forecast_hour,
        errors.join(" | ")
    )))
}

pub fn fetch_bytes_with_cache(
    fetch: &FetchRequest,
    cache_root: &std::path::Path,
    use_cache: bool,
) -> Result<CachedFetchResult, IoError> {
    if use_cache {
        if let Some(cached) = load_cached_fetch(cache_root, fetch)? {
            return Ok(cached);
        }
        if let Some(cached) = load_cached_raw_full_fetch(cache_root, fetch)? {
            return Ok(cached);
        }
    }
    if use_cache {
        let _cache_lock = acquire_fetch_cache_lock(cache_root, fetch)?;
        if let Some(cached) = load_cached_fetch(cache_root, fetch)? {
            return Ok(cached);
        }
        if let Some(cached) = load_cached_raw_full_fetch(cache_root, fetch)? {
            return Ok(cached);
        }
        if let Some(cached) = fetch_bytes_with_raw_full_cache(fetch, cache_root)? {
            return Ok(cached);
        }
        let result = fetch_bytes(fetch)?;
        store_cached_fetch(cache_root, fetch, &result)
    } else {
        let result = fetch_bytes(fetch)?;
        let (bytes_path, metadata_path) = fetch_cache_paths(cache_root, fetch);
        Ok(CachedFetchResult {
            result,
            cache_hit: false,
            bytes_path,
            metadata_path,
        })
    }
}

fn load_cached_raw_full_fetch(
    cache_root: &std::path::Path,
    fetch: &FetchRequest,
) -> Result<Option<CachedFetchResult>, IoError> {
    if !fetch_can_use_raw_full_file_cache(fetch) {
        return Ok(None);
    }
    for resolved in filtered_urls(fetch)? {
        if let Some(cached) =
            load_cached_raw_fetch(cache_root, resolved.source, &resolved.grib_url)?
        {
            return Ok(Some(cached));
        }
    }
    Ok(None)
}

fn fetch_bytes_with_raw_full_cache(
    fetch: &FetchRequest,
    cache_root: &std::path::Path,
) -> Result<Option<CachedFetchResult>, IoError> {
    if !fetch_can_use_raw_full_file_cache(fetch) {
        return Ok(None);
    }
    let client = client()?;
    let urls = filtered_urls(fetch)?;
    let patterns = fetch
        .variable_patterns
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let mut errors = Vec::new();
    for resolved in urls {
        if let Some(cached) =
            load_cached_raw_fetch(cache_root, resolved.source, &resolved.grib_url)?
        {
            return Ok(Some(cached));
        }
        let _raw_lock =
            acquire_raw_fetch_cache_lock(cache_root, resolved.source, &resolved.grib_url)?;
        if let Some(cached) =
            load_cached_raw_fetch(cache_root, resolved.source, &resolved.grib_url)?
        {
            return Ok(Some(cached));
        }
        match try_fetch_one(&client, &resolved, &patterns) {
            Ok(bytes) => {
                let result = FetchResult {
                    source: resolved.source,
                    url: resolved.grib_url,
                    bytes,
                };
                return store_cached_raw_fetch(cache_root, fetch, &result).map(Some);
            }
            Err(err) => errors.push(format!("{}: {}", resolved.source, err)),
        }
    }

    Err(IoError::Download(format!(
        "all sources failed for {} f{:03}: {}",
        fetch.request.model,
        fetch.request.forecast_hour,
        errors.join(" | ")
    )))
}

fn fetch_can_use_raw_full_file_cache(fetch: &FetchRequest) -> bool {
    fetch.variable_patterns.is_empty() || matches!(fetch.source_override, Some(SourceId::Nomads))
}

struct FetchCacheLock {
    path: PathBuf,
    file: Option<File>,
}

impl Drop for FetchCacheLock {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

fn acquire_fetch_cache_lock(
    cache_root: &std::path::Path,
    fetch: &FetchRequest,
) -> Result<FetchCacheLock, IoError> {
    let (bytes_path, _) = fetch_cache_paths(cache_root, fetch);
    let lock_path = bytes_path.with_file_name("fetch.grib2.lock");
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent).map_err(|err| IoError::Cache(err.to_string()))?;
    }

    let started = Instant::now();
    loop {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(mut file) => {
                writeln!(
                    file,
                    "pid={} model={} date={} cycle={:02} forecast_hour={}",
                    std::process::id(),
                    fetch.request.model,
                    fetch.request.cycle.date_yyyymmdd,
                    fetch.request.cycle.hour_utc,
                    fetch.request.forecast_hour
                )
                .map_err(|err| IoError::Cache(err.to_string()))?;
                return Ok(FetchCacheLock {
                    path: lock_path,
                    file: Some(file),
                });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if is_stale_fetch_cache_lock(&lock_path) {
                    let _ = fs::remove_file(&lock_path);
                    continue;
                }
                if started.elapsed() > FETCH_CACHE_LOCK_WAIT_TIMEOUT {
                    return Err(IoError::Cache(format!(
                        "timed out waiting for fetch cache lock {}",
                        lock_path.display()
                    )));
                }
                thread::sleep(FETCH_CACHE_LOCK_RETRY_AFTER);
            }
            Err(err) => return Err(IoError::Cache(err.to_string())),
        }
    }
}

fn acquire_raw_fetch_cache_lock(
    cache_root: &std::path::Path,
    source: SourceId,
    resolved_url: &str,
) -> Result<FetchCacheLock, IoError> {
    let (bytes_path, _) = raw_fetch_cache_paths(cache_root, source, resolved_url);
    let lock_path = bytes_path.with_file_name("fetch.grib2.lock");
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent).map_err(|err| IoError::Cache(err.to_string()))?;
    }

    let started = Instant::now();
    loop {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(mut file) => {
                writeln!(
                    file,
                    "pid={} source={} url={}",
                    std::process::id(),
                    source,
                    resolved_url
                )
                .map_err(|err| IoError::Cache(err.to_string()))?;
                return Ok(FetchCacheLock {
                    path: lock_path,
                    file: Some(file),
                });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if is_stale_fetch_cache_lock(&lock_path) {
                    let _ = fs::remove_file(&lock_path);
                    continue;
                }
                if started.elapsed() > FETCH_CACHE_LOCK_WAIT_TIMEOUT {
                    return Err(IoError::Cache(format!(
                        "timed out waiting for raw fetch cache lock {}",
                        lock_path.display()
                    )));
                }
                thread::sleep(FETCH_CACHE_LOCK_RETRY_AFTER);
            }
            Err(err) => return Err(IoError::Cache(err.to_string())),
        }
    }
}

fn is_stale_fetch_cache_lock(lock_path: &Path) -> bool {
    if fetch_cache_lock_pid_is_dead(lock_path) {
        return true;
    }

    fs::metadata(lock_path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age > FETCH_CACHE_LOCK_STALE_AFTER)
}

fn fetch_cache_lock_pid_is_dead(lock_path: &Path) -> bool {
    let Ok(contents) = fs::read_to_string(lock_path) else {
        return false;
    };
    let Some(pid) = contents.split_whitespace().find_map(|part| {
        part.strip_prefix("pid=")
            .and_then(|raw| raw.parse::<u32>().ok())
    }) else {
        return false;
    };
    if pid == std::process::id() {
        return false;
    }
    let proc_root = Path::new("/proc");
    proc_root.exists() && !proc_root.join(pid.to_string()).exists()
}

pub fn extract_field_from_bytes(
    bytes: &[u8],
    selector: FieldSelector,
) -> Result<SelectedField2D, IoError> {
    let mut fields = extract_fields_from_bytes(bytes, &[selector])?;
    debug_assert_eq!(fields.len(), 1);
    Ok(fields.swap_remove(0))
}

pub fn extract_fields_from_bytes(
    bytes: &[u8],
    selectors: &[FieldSelector],
) -> Result<Vec<SelectedField2D>, IoError> {
    let grib = Grib2File::from_bytes(bytes).map_err(|err| IoError::Grib(err.to_string()))?;
    extract_fields_from_grib2(&grib, selectors)
}

pub fn extract_field_from_grib2(
    grib: &Grib2File,
    selector: FieldSelector,
) -> Result<SelectedField2D, IoError> {
    let mut fields = extract_fields_from_grib2(grib, &[selector])?;
    debug_assert_eq!(fields.len(), 1);
    Ok(fields.swap_remove(0))
}

pub fn extract_fields_from_grib2(
    grib: &Grib2File,
    selectors: &[FieldSelector],
) -> Result<Vec<SelectedField2D>, IoError> {
    if selectors.is_empty() {
        return Ok(Vec::new());
    }

    let prepared = selectors
        .iter()
        .copied()
        .map(PreparedSelector::new)
        .collect::<Result<Vec<_>, _>>()?;
    let matched = match_prepared_selectors(grib, &prepared, None);

    let mut out = Vec::with_capacity(prepared.len());
    let mut grid_memo = GridMemo::new();
    for (prepared_selector, message) in prepared.iter().zip(matched.into_iter()) {
        let message = message
            .map(|(message, _)| message)
            .ok_or(IoError::FieldNotFound {
                selector: prepared_selector.selector,
            })?;
        out.push(build_selected_field(
            message,
            prepared_selector.selector,
            prepared_selector.selector.native_units(),
            &mut grid_memo,
        )?);
    }

    Ok(out)
}

/// Partial-success variant of `extract_fields_from_grib2`: selectors
/// whose GRIB message is absent from the file are returned in the
/// `missing` vector instead of erroring out. Callers that want per-
/// selector soft-fail (e.g. direct_batch, which renders many recipes
/// from one fetch and shouldn't abort the whole batch when one
/// selector is missing) opt into this variant; everyone else keeps
/// getting strict all-or-nothing semantics from the original function.
///
/// The only `Err` path here is a genuinely malformed selector or a
/// decode error on a matched message — neither of which is the "this
/// model doesn't expose that field at init time" case that the strict
/// variant treats identically.
pub fn extract_fields_from_grib2_partial(
    grib: &Grib2File,
    selectors: &[FieldSelector],
) -> Result<PartialExtraction, IoError> {
    extract_fields_from_grib2_partial_inner(grib, selectors, None)
}

pub fn extract_fields_from_grib2_partial_at_forecast_hour(
    grib: &Grib2File,
    selectors: &[FieldSelector],
    forecast_hour: u16,
) -> Result<PartialExtraction, IoError> {
    extract_fields_from_grib2_partial_inner(grib, selectors, Some(forecast_hour))
}

fn extract_fields_from_grib2_partial_inner(
    grib: &Grib2File,
    selectors: &[FieldSelector],
    forecast_hour: Option<u16>,
) -> Result<PartialExtraction, IoError> {
    let mut extracted = Vec::new();
    let mut missing = Vec::new();

    if selectors.is_empty() {
        return Ok(PartialExtraction { extracted, missing });
    }

    let prepared = selectors
        .iter()
        .copied()
        .map(PreparedSelector::new)
        .collect::<Result<Vec<_>, _>>()?;
    let matched = match_prepared_selectors(grib, &prepared, forecast_hour);

    let mut grid_memo = GridMemo::new();
    for (prepared_selector, message) in prepared.iter().zip(matched.into_iter()) {
        match message {
            Some((message, _)) => extracted.push(build_selected_field(
                message,
                prepared_selector.selector,
                prepared_selector.selector.native_units(),
                &mut grid_memo,
            )?),
            None => missing.push(prepared_selector.selector),
        }
    }

    Ok(PartialExtraction { extracted, missing })
}

/// The selector-to-message matching loop shared by every extraction lane:
/// for each prepared selector, the best-scoring matching message (lower
/// score wins, first wins ties via strict `<`).
fn match_prepared_selectors<'a>(
    grib: &'a Grib2File,
    prepared: &[PreparedSelector],
    forecast_hour: Option<u16>,
) -> Vec<Option<(&'a Grib2Message, u8)>> {
    let mut matched: Vec<Option<(&Grib2Message, u8)>> = vec![None; prepared.len()];
    for message in &grib.messages {
        for (index, prepared_selector) in prepared.iter().enumerate() {
            if prepared_selector.message.matches(message) {
                let Some(score) = prepared_selector.match_score(message, forecast_hour) else {
                    continue;
                };
                let replace = matched[index]
                    .map(|(_, best_score)| score < best_score)
                    .unwrap_or(true);
                if replace {
                    matched[index] = Some((message, score));
                }
            }
        }
    }
    matched
}

/// Result of a partial extraction: every selector the GRIB file served
/// in `extracted`, every selector whose message was absent in `missing`.
#[derive(Debug, Clone)]
pub struct PartialExtraction {
    pub extracted: Vec<SelectedField2D>,
    pub missing: Vec<FieldSelector>,
}

/// One extracted field as bare values: the selector, its native units, the
/// values after the exact normalization sequence the `SelectedField2D`
/// lane applies (unpack, alternating-i scan, row flip, per-row longitude
/// rotation, f64 -> f32), and the index of its shared coordinate grid in
/// [`PartialValuesExtraction::grids`].
#[derive(Debug, Clone)]
pub struct ExtractedFieldValues {
    pub selector: FieldSelector,
    pub units: String,
    pub values: Vec<f32>,
    /// Index into [`PartialValuesExtraction::grids`].
    pub grid_index: usize,
}

/// One distinct coordinate grid realized by a values-only extraction —
/// bit-identical to the `grid` every `SelectedField2D` sharing the same
/// GRIB `GridDefinition` would have carried, materialized once instead of
/// cloned per field.
#[derive(Debug, Clone)]
pub struct SharedExtractionGrid {
    pub grid: LatLonGrid,
    pub projection: Option<GridProjection>,
}

/// Values-only sibling of [`PartialExtraction`] for callers (the store
/// ingest) that hold many fields from one file at once: per-field values
/// plus each distinct grid exactly once. A full HRRR `prs` extraction
/// carries ~195 fields on one 15 MB grid — the per-field clones of the
/// `SelectedField2D` shape cost ~2.8 GB at peak, this shape costs 15 MB.
#[derive(Debug, Clone)]
pub struct PartialValuesExtraction {
    pub extracted: Vec<ExtractedFieldValues>,
    pub missing: Vec<FieldSelector>,
    pub grids: Vec<SharedExtractionGrid>,
}

pub fn extract_fields_partial_from_model_bytes(
    model: ModelId,
    bytes: &[u8],
    preferred_path: Option<&Path>,
    selectors: &[FieldSelector],
) -> Result<PartialExtraction, IoError> {
    extract_fields_partial_from_model_bytes_at_forecast_hour(
        model,
        bytes,
        preferred_path,
        selectors,
        None,
    )
}

pub fn extract_fields_partial_from_model_bytes_at_forecast_hour(
    model: ModelId,
    bytes: &[u8],
    preferred_path: Option<&Path>,
    selectors: &[FieldSelector],
    forecast_hour: Option<u16>,
) -> Result<PartialExtraction, IoError> {
    match model {
        ModelId::WrfGdex => extract_wrf_gdex_fields_partial(bytes, preferred_path, selectors),
        _ => {
            let grib =
                Grib2File::from_bytes(bytes).map_err(|err| IoError::Grib(err.to_string()))?;
            let mut partial = if let Some(forecast_hour) = forecast_hour {
                extract_fields_from_grib2_partial_at_forecast_hour(&grib, selectors, forecast_hour)?
            } else {
                extract_fields_from_grib2_partial(&grib, selectors)?
            };
            if model == ModelId::Nbm {
                synthesize_nbm_10m_wind_components_from_speed_direction(&grib, &mut partial)?;
            }
            Ok(partial)
        }
    }
}

/// Values-only sibling of
/// [`extract_fields_partial_from_model_bytes_at_forecast_hour`]: the same
/// model dispatch, the same selector matching and message scoring, the same
/// per-field value normalization — but each distinct coordinate grid is
/// returned exactly once in [`PartialValuesExtraction::grids`] instead of
/// being cloned into every field.
pub fn extract_field_values_partial_from_model_bytes_at_forecast_hour(
    model: ModelId,
    bytes: &[u8],
    _preferred_path: Option<&Path>,
    selectors: &[FieldSelector],
    forecast_hour: Option<u16>,
) -> Result<PartialValuesExtraction, IoError> {
    match model {
        ModelId::WrfGdex => Err(IoError::Wrf(
            "WRF/GDEX NetCDF support is not available in this build".to_string(),
        )),
        _ => {
            let grib =
                Grib2File::from_bytes(bytes).map_err(|err| IoError::Grib(err.to_string()))?;
            let mut extracted = Vec::new();
            let mut missing = Vec::new();
            let mut grid_memo = GridMemo::new();
            if !selectors.is_empty() {
                let prepared = selectors
                    .iter()
                    .copied()
                    .map(PreparedSelector::new)
                    .collect::<Result<Vec<_>, _>>()?;
                let matched = match_prepared_selectors(&grib, &prepared, forecast_hour);
                for (prepared_selector, message) in prepared.iter().zip(matched.into_iter()) {
                    match message {
                        Some((message, _)) => extracted.push(build_field_values(
                            message,
                            prepared_selector.selector,
                            prepared_selector.selector.native_units(),
                            &mut grid_memo,
                        )?),
                        None => missing.push(prepared_selector.selector),
                    }
                }
            }
            if model == ModelId::Nbm {
                synthesize_nbm_10m_wind_component_values_from_speed_direction(
                    &grib,
                    &mut extracted,
                    &mut missing,
                    &mut grid_memo,
                )?;
            }
            Ok(PartialValuesExtraction {
                extracted,
                missing,
                grids: grid_memo.into_shared_grids(),
            })
        }
    }
}

/// Values-lane twin of
/// [`synthesize_nbm_10m_wind_components_from_speed_direction`]: identical
/// message selection and u/v arithmetic, shared-grid output shape.
fn synthesize_nbm_10m_wind_component_values_from_speed_direction(
    grib: &Grib2File,
    extracted: &mut Vec<ExtractedFieldValues>,
    missing: &mut Vec<FieldSelector>,
    grid_memo: &mut GridMemo,
) -> Result<(), IoError> {
    let u_selector = FieldSelector::height_agl(CanonicalField::UWind, 10);
    let v_selector = FieldSelector::height_agl(CanonicalField::VWind, 10);
    let needs_u = missing.contains(&u_selector);
    let needs_v = missing.contains(&v_selector);
    if !needs_u && !needs_v {
        return Ok(());
    }

    let speed_selector = StructuredMessageSelector {
        parameters: PARAMETER_WIND_SPEED,
        level: LevelMatch::HeightAboveGroundMeters(10),
        units: "m/s",
    };
    let direction_selector = StructuredMessageSelector {
        parameters: PARAMETER_WIND_DIRECTION,
        level: LevelMatch::HeightAboveGroundMeters(10),
        units: "deg",
    };
    let Some(speed_message) = grib
        .messages
        .iter()
        .find(|message| speed_selector.matches(message))
    else {
        return Ok(());
    };
    let Some(direction_message) = grib
        .messages
        .iter()
        .find(|message| direction_selector.matches(message))
    else {
        return Ok(());
    };

    let speed = build_field_values(speed_message, u_selector, speed_selector.units, grid_memo)?;
    let direction = build_field_values(
        direction_message,
        v_selector,
        direction_selector.units,
        grid_memo,
    )?;
    let speed_shape = grid_memo.slots[speed.grid_index].0.grid.shape;
    let direction_shape = grid_memo.slots[direction.grid_index].0.grid.shape;
    if speed_shape != direction_shape || speed.values.len() != direction.values.len() {
        return Ok(());
    }

    let mut u_values = Vec::with_capacity(speed.values.len());
    let mut v_values = Vec::with_capacity(speed.values.len());
    for (speed_ms, direction_deg) in speed.values.iter().zip(direction.values.iter()) {
        if speed_ms.is_finite() && direction_deg.is_finite() {
            let theta = f64::from(*direction_deg).to_radians();
            u_values.push((-f64::from(*speed_ms) * theta.sin()) as f32);
            v_values.push((-f64::from(*speed_ms) * theta.cos()) as f32);
        } else {
            u_values.push(f32::NAN);
            v_values.push(f32::NAN);
        }
    }

    if needs_u {
        extracted.push(ExtractedFieldValues {
            selector: u_selector,
            units: "m/s".to_string(),
            values: u_values,
            grid_index: speed.grid_index,
        });
    }
    if needs_v {
        extracted.push(ExtractedFieldValues {
            selector: v_selector,
            units: "m/s".to_string(),
            values: v_values,
            grid_index: speed.grid_index,
        });
    }
    missing.retain(|selector| *selector != u_selector && *selector != v_selector);
    Ok(())
}

fn synthesize_nbm_10m_wind_components_from_speed_direction(
    grib: &Grib2File,
    partial: &mut PartialExtraction,
) -> Result<(), IoError> {
    let u_selector = FieldSelector::height_agl(CanonicalField::UWind, 10);
    let v_selector = FieldSelector::height_agl(CanonicalField::VWind, 10);
    let needs_u = partial.missing.contains(&u_selector);
    let needs_v = partial.missing.contains(&v_selector);
    if !needs_u && !needs_v {
        return Ok(());
    }

    let speed_selector = StructuredMessageSelector {
        parameters: PARAMETER_WIND_SPEED,
        level: LevelMatch::HeightAboveGroundMeters(10),
        units: "m/s",
    };
    let direction_selector = StructuredMessageSelector {
        parameters: PARAMETER_WIND_DIRECTION,
        level: LevelMatch::HeightAboveGroundMeters(10),
        units: "deg",
    };
    let Some(speed_message) = grib
        .messages
        .iter()
        .find(|message| speed_selector.matches(message))
    else {
        return Ok(());
    };
    let Some(direction_message) = grib
        .messages
        .iter()
        .find(|message| direction_selector.matches(message))
    else {
        return Ok(());
    };

    let mut grid_memo = GridMemo::new();
    let speed = build_selected_field(
        speed_message,
        u_selector,
        speed_selector.units,
        &mut grid_memo,
    )?;
    let direction = build_selected_field(
        direction_message,
        v_selector,
        direction_selector.units,
        &mut grid_memo,
    )?;
    if speed.grid.shape != direction.grid.shape || speed.values.len() != direction.values.len() {
        return Ok(());
    }

    let mut u_values = Vec::with_capacity(speed.values.len());
    let mut v_values = Vec::with_capacity(speed.values.len());
    for (speed_ms, direction_deg) in speed.values.iter().zip(direction.values.iter()) {
        if speed_ms.is_finite() && direction_deg.is_finite() {
            let theta = f64::from(*direction_deg).to_radians();
            u_values.push((-f64::from(*speed_ms) * theta.sin()) as f32);
            v_values.push((-f64::from(*speed_ms) * theta.cos()) as f32);
        } else {
            u_values.push(f32::NAN);
            v_values.push(f32::NAN);
        }
    }

    if needs_u {
        let mut u = SelectedField2D::new(u_selector, "m/s", speed.grid.clone(), u_values)?;
        if let Some(projection) = speed.projection.clone() {
            u = u.with_projection(projection);
        }
        partial.extracted.push(u);
    }
    if needs_v {
        let mut v = SelectedField2D::new(v_selector, "m/s", speed.grid.clone(), v_values)?;
        if let Some(projection) = speed.projection.clone() {
            v = v.with_projection(projection);
        }
        partial.extracted.push(v);
    }
    partial
        .missing
        .retain(|selector| *selector != u_selector && *selector != v_selector);
    Ok(())
}

/// `ModelId::WrfGdex` is still a registered model (URL builders and recipes
/// reference it), so the extraction dispatch needs this arm even though
/// rusty-weather ships without the NetCDF/WRF decode path.
fn extract_wrf_gdex_fields_partial(
    _bytes: &[u8],
    _preferred_path: Option<&Path>,
    _selectors: &[FieldSelector],
) -> Result<PartialExtraction, IoError> {
    Err(IoError::Wrf(
        "WRF/GDEX NetCDF support is not available in this build".to_string(),
    ))
}

pub fn extract_pressure_field_from_bytes(
    bytes: &[u8],
    field: CanonicalField,
    level_hpa: u16,
) -> Result<SelectedField2D, IoError> {
    extract_field_from_bytes(bytes, FieldSelector::isobaric(field, level_hpa))
}

pub fn extract_pressure_field_from_grib2(
    grib: &Grib2File,
    field: CanonicalField,
    level_hpa: u16,
) -> Result<SelectedField2D, IoError> {
    extract_field_from_grib2(grib, FieldSelector::isobaric(field, level_hpa))
}

pub const HRRR_WRFNAT_HYBRID_LEVEL_COUNT: u16 = 50;

#[derive(Debug, Clone, PartialEq)]
pub struct HrrrWrfnatSmokeExtraction {
    pub hybrid_smoke: SelectedHybridLevelVolume,
    pub hybrid_pressure: SelectedHybridLevelVolume,
    pub near_surface_smoke: SelectedField2D,
    pub column_smoke: SelectedField2D,
}

pub fn hrrr_wrfnat_hybrid_levels() -> Vec<u16> {
    (1..=HRRR_WRFNAT_HYBRID_LEVEL_COUNT).collect()
}

pub fn extract_hybrid_level_volume_from_bytes(
    bytes: &[u8],
    field: CanonicalField,
    levels_hybrid: &[u16],
) -> Result<SelectedHybridLevelVolume, IoError> {
    let grib = Grib2File::from_bytes(bytes).map_err(|err| IoError::Grib(err.to_string()))?;
    extract_hybrid_level_volume_from_grib2(&grib, field, levels_hybrid)
}

pub fn extract_hybrid_level_volume_from_grib2(
    grib: &Grib2File,
    field: CanonicalField,
    levels_hybrid: &[u16],
) -> Result<SelectedHybridLevelVolume, IoError> {
    let selectors = levels_hybrid
        .iter()
        .copied()
        .map(|level| FieldSelector::hybrid_level(field, level))
        .collect::<Vec<_>>();
    let slices = extract_fields_from_grib2(grib, &selectors)?;
    build_hybrid_level_volume(field, levels_hybrid, slices)
}

pub fn extract_hrrr_wrfnat_smoke_fields_from_bytes(
    bytes: &[u8],
) -> Result<HrrrWrfnatSmokeExtraction, IoError> {
    let grib = Grib2File::from_bytes(bytes).map_err(|err| IoError::Grib(err.to_string()))?;
    extract_hrrr_wrfnat_smoke_fields_from_grib2(&grib)
}

pub fn extract_hrrr_wrfnat_smoke_fields_from_grib2(
    grib: &Grib2File,
) -> Result<HrrrWrfnatSmokeExtraction, IoError> {
    let levels = hrrr_wrfnat_hybrid_levels();
    let hybrid_smoke =
        extract_hybrid_level_volume_from_grib2(grib, CanonicalField::SmokeMassDensity, &levels)?;
    let hybrid_pressure =
        extract_hybrid_level_volume_from_grib2(grib, CanonicalField::Pressure, &levels)?;
    let mut smoke_maps = extract_fields_from_grib2(
        grib,
        &[
            FieldSelector::height_agl(CanonicalField::SmokeMassDensity, 8),
            FieldSelector::entire_atmosphere(CanonicalField::ColumnIntegratedSmoke),
        ],
    )?;
    debug_assert_eq!(smoke_maps.len(), 2);
    let column_smoke = smoke_maps
        .pop()
        .expect("column smoke selector should be present after successful extraction");
    let near_surface_smoke = smoke_maps
        .pop()
        .expect("near-surface smoke selector should be present after successful extraction");

    Ok(HrrrWrfnatSmokeExtraction {
        hybrid_smoke,
        hybrid_pressure,
        near_surface_smoke,
        column_smoke,
    })
}

fn build_hybrid_level_volume(
    field: CanonicalField,
    levels_hybrid: &[u16],
    slices: Vec<SelectedField2D>,
) -> Result<SelectedHybridLevelVolume, IoError> {
    let Some(first) = slices.first() else {
        return Err(rustwx_core::RustwxError::EmptyHybridLevels.into());
    };

    let expected_grid = first.grid.clone();
    let expected_units = first.units.clone();
    let expected_projection = first.projection.clone();

    for slice in &slices {
        if slice.grid != expected_grid {
            return Err(IoError::Grib(format!(
                "hybrid volume for field '{field}' used inconsistent grids across levels"
            )));
        }
        if slice.units != expected_units {
            return Err(IoError::Grib(format!(
                "hybrid volume for field '{field}' used inconsistent units across levels"
            )));
        }
        if slice.projection != expected_projection {
            return Err(IoError::Grib(format!(
                "hybrid volume for field '{field}' used inconsistent projections across levels"
            )));
        }
    }

    let values = slices
        .into_iter()
        .flat_map(|slice| slice.values)
        .collect::<Vec<_>>();
    let mut volume = SelectedHybridLevelVolume::new(
        field,
        levels_hybrid.to_vec(),
        expected_units,
        expected_grid,
        values,
    )?;
    if let Some(projection) = expected_projection {
        volume = volume.with_projection(projection);
    }
    Ok(volume)
}

fn filtered_urls(fetch: &FetchRequest) -> Result<Vec<ResolvedUrl>, IoError> {
    let urls = resolve_urls(&fetch.request)?;
    Ok(match fetch.source_override {
        Some(source) => urls
            .into_iter()
            .filter(|url| url.source == source)
            .collect(),
        None => urls,
    })
}

fn fetch_request_is_available(
    client: &DownloadClient,
    fetch: &FetchRequest,
) -> Result<bool, IoError> {
    let urls = filtered_urls(fetch)?;
    Ok(any_source_available(&urls, |resolved| {
        probe_availability(client, resolved)
    }))
}

fn probe_availability(client: &DownloadClient, resolved: &ResolvedUrl) -> bool {
    if matches!(resolved.source, SourceId::Nomads) {
        client.get_range(&resolved.grib_url, 0, 0).is_ok()
    } else {
        client.head_ok(resolved.availability_probe_url())
    }
}

fn any_source_available<F>(resolved: &[ResolvedUrl], mut probe: F) -> bool
where
    F: FnMut(&ResolvedUrl) -> bool,
{
    resolved.iter().any(&mut probe)
}

fn should_parallelize_hour_availability_probes(
    source_override: Option<SourceId>,
    summary: &rustwx_models::ModelSummary,
) -> bool {
    match source_override {
        Some(source) => !matches!(source, SourceId::Nomads),
        None => summary
            .sources
            .iter()
            .all(|source| source.id != SourceId::Nomads),
    }
}

fn try_fetch_one(
    client: &DownloadClient,
    resolved: &ResolvedUrl,
    variable_patterns: &[&str],
) -> Result<Vec<u8>, String> {
    if resolved.source == SourceId::Nomads {
        let bytes = client
            .get_bytes(&resolved.grib_url)
            .map_err(|err| err.to_string())?;
        return maybe_decompress_grib_payload(&resolved.grib_url, bytes);
    }

    if should_use_idx_subset_fetch(resolved.source) && !variable_patterns.is_empty() {
        if let Some(idx_url) = &resolved.idx_url {
            if let Ok(idx_text) = client.get_text(idx_url) {
                if let Some(ranges) = idx_subset_ranges(&idx_text, variable_patterns)? {
                    let bytes = client
                        .get_ranges(&resolved.grib_url, &ranges)
                        .map_err(|err| err.to_string())?;
                    return maybe_decompress_grib_payload(&resolved.grib_url, bytes);
                }
            }
        }
    }
    let result = if should_use_parallel_whole_file_fetch(resolved.source) {
        client.get_bytes_parallel_whole(&resolved.grib_url)
    } else {
        client.get_bytes(&resolved.grib_url)
    };
    let bytes = result.map_err(|err| err.to_string())?;
    maybe_decompress_grib_payload(&resolved.grib_url, bytes)
}

fn maybe_decompress_grib_payload(url: &str, bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    let is_gzip = url.to_ascii_lowercase().ends_with(".gz")
        || bytes
            .get(0..2)
            .is_some_and(|magic| magic == [0x1f_u8, 0x8b_u8]);
    if !is_gzip {
        return Ok(bytes);
    }

    let mut decoder = flate2::read::GzDecoder::new(bytes.as_slice());
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|err| format!("gzip decompress {url}: {err}"))?;
    Ok(decompressed)
}

fn should_use_parallel_whole_file_fetch(source: SourceId) -> bool {
    matches!(source, SourceId::Aws | SourceId::Google)
}

fn should_use_idx_subset_fetch(source: SourceId) -> bool {
    // NOMADS production fetches full GRIB files. The .idx sidecar is allowed
    // for availability probes only, not product subsetting.
    matches!(source, SourceId::Aws | SourceId::Google)
}

fn idx_subset_ranges(idx_text: &str, patterns: &[&str]) -> Result<Option<Vec<(u64, u64)>>, String> {
    let entries = parse_idx(idx_text);
    if entries.is_empty() {
        return Ok(None);
    }

    let mut selected = Vec::new();
    let mut seen_offsets = HashSet::new();
    for pattern in patterns {
        for entry in find_entries(&entries, pattern) {
            if seen_offsets.insert(entry.byte_offset) {
                selected.push(entry);
            }
        }
    }

    if selected.is_empty() {
        return Ok(None);
    }
    Ok(Some(coalesce_contiguous_ranges(byte_ranges(
        &entries, &selected,
    ))))
}

fn coalesce_contiguous_ranges(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    if ranges.len() <= 1 {
        return ranges;
    }
    ranges.sort_unstable_by_key(|range| range.0);

    let mut merged = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        let Some((_, last_end)) = merged.last_mut() else {
            merged.push((start, end));
            continue;
        };
        if *last_end != u64::MAX && start <= last_end.saturating_add(1) {
            *last_end = (*last_end).max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

fn candidate_hours(model: ModelId, cycle_hour: u8) -> Vec<u16> {
    // Delegate to the canonical schedule in rustwx-models so availability
    // probes match the cycle-aware horizons that the catalog and fetch
    // plan already encode (e.g. ECMWF 00/12z goes to 360h, 06/18z to 144h).
    rustwx_models::supported_forecast_hours(model, cycle_hour)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParameterCode {
    discipline: u8,
    category: u8,
    number: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LevelMatch {
    Surface,
    MeanSeaLevel,
    AltitudeMeters(u16),
    IsobaricHpa(u16),
    HybridLevel(u16),
    EntireAtmosphere,
    NominalTop,
    ExactLevelType(u8),
    HeightAboveGroundMeters(u16),
    SurfaceOrHeightAboveGroundMeters(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StructuredMessageSelector {
    parameters: &'static [ParameterCode],
    level: LevelMatch,
    units: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct PreparedSelector {
    selector: FieldSelector,
    message: StructuredMessageSelector,
}

const PARAMETER_HGT: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 3,
    number: 5,
}];
const PARAMETER_PRESSURE: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 3,
    number: 0,
}];
const PARAMETER_TMP: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 0,
    number: 0,
}];
const PARAMETER_DPT: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 0,
    number: 6,
}];
const PARAMETER_RH: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 1,
    number: 1,
}];
const PARAMETER_PWAT: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 1,
    number: 3,
}];
const PARAMETER_TOTAL_PRECIPITATION: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 1,
    number: 8,
}];
const PARAMETER_PROBABILITY_OF_PRECIPITATION: &[ParameterCode] = PARAMETER_TOTAL_PRECIPITATION;
const PARAMETER_CATEGORICAL_RAIN: &[ParameterCode] = &[
    ParameterCode {
        discipline: 0,
        category: 1,
        number: 192,
    },
    ParameterCode {
        discipline: 0,
        category: 1,
        number: 33,
    },
];
const PARAMETER_CATEGORICAL_FREEZING_RAIN: &[ParameterCode] = &[
    ParameterCode {
        discipline: 0,
        category: 1,
        number: 193,
    },
    ParameterCode {
        discipline: 0,
        category: 1,
        number: 34,
    },
];
const PARAMETER_CATEGORICAL_ICE_PELLETS: &[ParameterCode] = &[
    ParameterCode {
        discipline: 0,
        category: 1,
        number: 194,
    },
    ParameterCode {
        discipline: 0,
        category: 1,
        number: 35,
    },
];
const PARAMETER_CATEGORICAL_SNOW: &[ParameterCode] = &[
    ParameterCode {
        discipline: 0,
        category: 1,
        number: 195,
    },
    ParameterCode {
        discipline: 0,
        category: 1,
        number: 36,
    },
];
const PARAMETER_UGRD: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 2,
    number: 2,
}];
const PARAMETER_VGRD: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 2,
    number: 3,
}];
const PARAMETER_WIND_DIRECTION: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 2,
    number: 0,
}];
const PARAMETER_WIND_SPEED: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 2,
    number: 1,
}];
const PARAMETER_WIND_GUST: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 2,
    number: 22,
}];
// Only absolute vorticity is wired right now. Relative vorticity needs its own
// explicit selector and GRIB parameter mapping before it should be exposed.
const PARAMETER_ABSOLUTE_VORTICITY: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 2,
    number: 10,
}];
const PARAMETER_MSLP: &[ParameterCode] = &[
    ParameterCode {
        discipline: 0,
        category: 3,
        number: 0,
    },
    ParameterCode {
        discipline: 0,
        category: 3,
        number: 1,
    },
    ParameterCode {
        discipline: 0,
        category: 3,
        number: 192,
    },
    ParameterCode {
        discipline: 0,
        category: 3,
        number: 198,
    },
];
const PARAMETER_LANDSEA_MASK: &[ParameterCode] = &[ParameterCode {
    discipline: 2,
    category: 0,
    number: 0,
}];
const PARAMETER_TOTAL_CLOUD_COVER: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 6,
    number: 1,
}];
const PARAMETER_LOW_CLOUD_COVER: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 6,
    number: 3,
}];
const PARAMETER_MIDDLE_CLOUD_COVER: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 6,
    number: 4,
}];
const PARAMETER_HIGH_CLOUD_COVER: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 6,
    number: 5,
}];
const PARAMETER_VISIBILITY: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 19,
    number: 0,
}];
const PARAMETER_SIMULATED_IR: &[ParameterCode] = &[ParameterCode {
    discipline: 3,
    category: 192,
    number: 7,
}];
const PARAMETER_RADAR_REFLECTIVITY: &[ParameterCode] = &[
    // MRMS ReflectivityAtLowestAltitude.
    ParameterCode {
        discipline: 209,
        category: 3,
        number: 57,
    },
    ParameterCode {
        discipline: 0,
        category: 16,
        number: 4,
    },
    ParameterCode {
        discipline: 0,
        category: 16,
        number: 195,
    },
];
const PARAMETER_COMPOSITE_REFLECTIVITY: &[ParameterCode] = &[
    // MRMS MergedReflectivityQCComposite.
    ParameterCode {
        discipline: 209,
        category: 10,
        number: 0,
    },
    ParameterCode {
        discipline: 0,
        category: 16,
        number: 196,
    },
    ParameterCode {
        discipline: 0,
        category: 16,
        number: 5,
    },
    ParameterCode {
        discipline: 0,
        category: 1,
        number: 209,
    },
];
const PARAMETER_UPDRAFT_HELICITY: &[ParameterCode] = &[
    ParameterCode {
        discipline: 0,
        category: 7,
        number: 199,
    },
    ParameterCode {
        discipline: 0,
        category: 7,
        number: 15,
    },
];
const PARAMETER_SMOKE_MASS_DENSITY: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 20,
    number: 0,
}];
const PARAMETER_COLUMN_INTEGRATED_SMOKE: &[ParameterCode] = &[ParameterCode {
    discipline: 0,
    category: 20,
    number: 1,
}];

impl StructuredMessageSelector {
    fn matches(self, message: &Grib2Message) -> bool {
        self.parameters.iter().any(|parameter| {
            message.discipline == parameter.discipline
                && message.product.parameter_category == parameter.category
                && message.product.parameter_number == parameter.number
        }) && self.level.matches(message)
    }
}

impl PreparedSelector {
    fn new(selector: FieldSelector) -> Result<Self, IoError> {
        Ok(Self {
            selector,
            message: StructuredMessageSelector::try_from(selector)?,
        })
    }

    fn match_score(self, message: &Grib2Message, forecast_hour: Option<u16>) -> Option<u8> {
        let product_score = product_template_match_score(self.selector, message)?;
        let forecast_score = if let Some(forecast_hour) = forecast_hour {
            forecast_hour_match_score(message, forecast_hour)?
        } else {
            0
        };
        Some(product_score.saturating_add(forecast_score))
    }
}

fn forecast_hour_match_score(message: &Grib2Message, expected_hour: u16) -> Option<u8> {
    let start_hour = time_value_to_hours(
        message.product.time_range_unit,
        message.product.forecast_time,
    )?;
    if start_hour == expected_hour as u32 {
        return Some(0);
    }
    let end_hour = message
        .product
        .statistical_time_range_hours()
        .map(|length| start_hour.saturating_add(length as u32));
    if end_hour == Some(expected_hour as u32) {
        return Some(1);
    }
    None
}

fn time_value_to_hours(unit: u8, value: u32) -> Option<u32> {
    match unit {
        // WMO Code Table 4.4: minute, hour, day, 3 hours, 6 hours, 12 hours.
        0 => (value % 60 == 0).then_some(value / 60),
        1 => Some(value),
        2 => value.checked_mul(24),
        10 => value.checked_mul(3),
        11 => value.checked_mul(6),
        12 => value.checked_mul(12),
        _ => None,
    }
}

fn product_template_match_score(selector: FieldSelector, message: &Grib2Message) -> Option<u8> {
    match selector.product {
        FieldProduct::Default => default_product_template_match_score(selector, message),
        FieldProduct::EnsembleMean => derived_forecast_match_score(message, &[0, 1]),
        FieldProduct::EnsembleStandardDeviation => derived_forecast_match_score(message, &[2, 3]),
        FieldProduct::EnsembleSpread => derived_forecast_match_score(message, &[4]),
        FieldProduct::EnsembleMinimum => derived_forecast_match_score(message, &[8]),
        FieldProduct::EnsembleMaximum => derived_forecast_match_score(message, &[9]),
        FieldProduct::Percentile(percentile) => percentile_product_match_score(message, percentile),
        FieldProduct::Probability(selection) => probability_product_match_score(message, selection),
    }
}

fn default_product_template_match_score(
    selector: FieldSelector,
    message: &Grib2Message,
) -> Option<u8> {
    if selector.field == CanonicalField::ProbabilityOfPrecipitation {
        return if is_probability_product_template(message.product.template) {
            Some(0)
        } else {
            None
        };
    }

    if selector.field == CanonicalField::TotalPrecipitation {
        return match message.product.template {
            8 | 11 | 12 if message.product.derived_forecast_type.is_none() => {
                // A surface file may carry BOTH the run-total (0→h) and the
                // trailing-window ((h−1)→h) accumulation; both end at hour h
                // and tie on the end-hour forecast score, so without this the
                // winner is file order (HRRR puts the run total first — by
                // luck correct; RRFS-A puts the window first, which silently
                // stored the 1 h window as `apcp_run_total`, caught live on
                // f002 2026-06-11). Prefer the accumulation that starts at
                // the run start. The trailing-window selection is unaffected:
                // it re-selects at h−1, where only the window's start hour
                // matches (the run total's start and end both miss) — the
                // start-mismatch penalty still leaves it the only candidate.
                let starts_at_run_start = time_value_to_hours(
                    message.product.time_range_unit,
                    message.product.forecast_time,
                ) == Some(0);
                Some(if starts_at_run_start { 0 } else { 2 })
            }
            8 | 11 | 12 if matches!(message.product.derived_forecast_type, Some(0) | Some(1)) => {
                Some(20)
            }
            0 | 1 => Some(10),
            _ => None,
        };
    }

    if is_probability_product_template(message.product.template)
        || is_percentile_product_template(message.product.template)
    {
        return None;
    }
    if message.product.derived_forecast_type.is_some() {
        return matches!(message.product.derived_forecast_type, Some(0) | Some(1)).then_some(20);
    }

    if !selector_prefers_instantaneous_message(selector) {
        return Some(0);
    }

    match message.product.template {
        0 => Some(0),
        1 => Some(1),
        8 | 11 | 12 => Some(10),
        _ => None,
    }
}

fn is_probability_product_template(template: u16) -> bool {
    matches!(template, 5 | 9)
}

fn is_percentile_product_template(template: u16) -> bool {
    matches!(template, 6 | 10)
}

fn derived_forecast_match_score(message: &Grib2Message, accepted_codes: &[u8]) -> Option<u8> {
    let code = message.product.derived_forecast_type?;
    accepted_codes.contains(&code).then_some(0)
}

fn percentile_product_match_score(message: &Grib2Message, percentile: u8) -> Option<u8> {
    if is_percentile_product_template(message.product.template)
        && message.product.percentile_value == Some(percentile)
    {
        return Some(0);
    }
    let derived_code = percentile_derived_forecast_code(percentile)?;
    (message.product.derived_forecast_type == Some(derived_code)).then_some(5)
}

fn percentile_derived_forecast_code(percentile: u8) -> Option<u8> {
    match percentile {
        5 => Some(201),
        10 => Some(193),
        25 => Some(202),
        50 => Some(194),
        75 => Some(203),
        90 => Some(195),
        95 => Some(204),
        _ => None,
    }
}

fn probability_product_match_score(
    message: &Grib2Message,
    selection: ProbabilitySelection,
) -> Option<u8> {
    if !is_probability_product_template(message.product.template) {
        return None;
    }
    if let Some(probability_type) = selection.probability_type {
        if message.product.probability_type != Some(probability_type) {
            return None;
        }
    }
    let (semantic_lower_limit, semantic_upper_limit) = probability_semantic_limits(message);
    if let Some(lower) = selection.lower_limit_milli {
        if semantic_lower_limit != Some(lower) {
            return None;
        }
    }
    if let Some(upper) = selection.upper_limit_milli {
        if semantic_upper_limit != Some(upper) {
            return None;
        }
    }
    Some(0)
}

fn probability_semantic_limits(message: &Grib2Message) -> (Option<i64>, Option<i64>) {
    let lower = scaled_limit_milli(message.product.probability_lower_limit);
    let upper = scaled_limit_milli(message.product.probability_upper_limit);
    match message.product.probability_type {
        // GRIB2 Code Table 4.9 stores "below lower limit" and "above upper limit" using
        // raw lower/upper slots, but rustwx selectors describe the meteorological threshold.
        Some(0) => (None, lower),
        Some(1) => (upper, None),
        Some(2) => (lower, upper),
        Some(3) => (lower, None),
        Some(4) => (None, upper),
        _ => (lower, upper),
    }
}

fn scaled_limit_milli(actual: Option<f64>) -> Option<i64> {
    actual.map(|actual| (actual * 1000.0).round() as i64)
}

fn selector_prefers_instantaneous_message(selector: FieldSelector) -> bool {
    !matches!(
        selector.field,
        CanonicalField::WindGust
            | CanonicalField::CategoricalRain
            | CanonicalField::CategoricalFreezingRain
            | CanonicalField::CategoricalIcePellets
            | CanonicalField::CategoricalSnow
    )
}

impl TryFrom<FieldSelector> for StructuredMessageSelector {
    type Error = IoError;

    fn try_from(selector: FieldSelector) -> Result<Self, Self::Error> {
        match selector {
            FieldSelector {
                field: CanonicalField::Pressure,
                vertical: VerticalSelector::Surface,
                ..
            } => Ok(Self {
                parameters: PARAMETER_PRESSURE,
                level: LevelMatch::Surface,
                units: "Pa",
            }),
            FieldSelector {
                field: CanonicalField::Pressure,
                vertical: VerticalSelector::HybridLevel(level),
                ..
            } if is_supported_hrrr_smoke_hybrid_level(level) => Ok(Self {
                parameters: PARAMETER_PRESSURE,
                level: LevelMatch::HybridLevel(level),
                units: "Pa",
            }),
            FieldSelector {
                field: CanonicalField::GeopotentialHeight,
                vertical: VerticalSelector::IsobaricHpa(level_hpa),
                ..
            } if is_supported_upper_air_level(level_hpa) => Ok(Self {
                parameters: PARAMETER_HGT,
                level: LevelMatch::IsobaricHpa(level_hpa),
                units: "gpm",
            }),
            FieldSelector {
                field: CanonicalField::GeopotentialHeight,
                vertical: VerticalSelector::Surface,
                ..
            } => Ok(Self {
                parameters: PARAMETER_HGT,
                level: LevelMatch::Surface,
                units: "gpm",
            }),
            FieldSelector {
                field: CanonicalField::Temperature,
                vertical: VerticalSelector::IsobaricHpa(level_hpa),
                ..
            } if is_supported_upper_air_level(level_hpa) => Ok(Self {
                parameters: PARAMETER_TMP,
                level: LevelMatch::IsobaricHpa(level_hpa),
                units: "K",
            }),
            FieldSelector {
                field: CanonicalField::RelativeHumidity,
                vertical: VerticalSelector::IsobaricHpa(level_hpa),
                ..
            } if is_supported_upper_air_level(level_hpa) => Ok(Self {
                parameters: PARAMETER_RH,
                level: LevelMatch::IsobaricHpa(level_hpa),
                units: "%",
            }),
            FieldSelector {
                field: CanonicalField::Dewpoint,
                vertical: VerticalSelector::IsobaricHpa(level_hpa),
                ..
            } if is_supported_upper_air_level(level_hpa) => Ok(Self {
                parameters: PARAMETER_DPT,
                level: LevelMatch::IsobaricHpa(level_hpa),
                units: "K",
            }),
            FieldSelector {
                field: CanonicalField::Temperature,
                vertical: VerticalSelector::HeightAboveGroundMeters(2),
                ..
            } => Ok(Self {
                parameters: PARAMETER_TMP,
                level: LevelMatch::HeightAboveGroundMeters(2),
                units: "K",
            }),
            FieldSelector {
                field: CanonicalField::Dewpoint,
                vertical: VerticalSelector::HeightAboveGroundMeters(2),
                ..
            } => Ok(Self {
                parameters: PARAMETER_DPT,
                level: LevelMatch::HeightAboveGroundMeters(2),
                units: "K",
            }),
            FieldSelector {
                field: CanonicalField::RelativeHumidity,
                vertical: VerticalSelector::HeightAboveGroundMeters(2),
                ..
            } => Ok(Self {
                parameters: PARAMETER_RH,
                level: LevelMatch::HeightAboveGroundMeters(2),
                units: "%",
            }),
            FieldSelector {
                field: CanonicalField::SmokeMassDensity,
                vertical: VerticalSelector::HybridLevel(level),
                ..
            } if is_supported_hrrr_smoke_hybrid_level(level) => Ok(Self {
                parameters: PARAMETER_SMOKE_MASS_DENSITY,
                level: LevelMatch::HybridLevel(level),
                units: "kg/m^3",
            }),
            FieldSelector {
                field: CanonicalField::AbsoluteVorticity,
                vertical: VerticalSelector::IsobaricHpa(level_hpa),
                ..
            } if is_supported_upper_air_level(level_hpa) => Ok(Self {
                parameters: PARAMETER_ABSOLUTE_VORTICITY,
                level: LevelMatch::IsobaricHpa(level_hpa),
                units: "s^-1",
            }),
            FieldSelector {
                field: CanonicalField::UWind,
                vertical: VerticalSelector::IsobaricHpa(level_hpa),
                ..
            } if is_supported_upper_air_level(level_hpa) => Ok(Self {
                parameters: PARAMETER_UGRD,
                level: LevelMatch::IsobaricHpa(level_hpa),
                units: "m/s",
            }),
            FieldSelector {
                field: CanonicalField::VWind,
                vertical: VerticalSelector::IsobaricHpa(level_hpa),
                ..
            } if is_supported_upper_air_level(level_hpa) => Ok(Self {
                parameters: PARAMETER_VGRD,
                level: LevelMatch::IsobaricHpa(level_hpa),
                units: "m/s",
            }),
            FieldSelector {
                field: CanonicalField::UWind,
                vertical: VerticalSelector::HeightAboveGroundMeters(10),
                ..
            } => Ok(Self {
                parameters: PARAMETER_UGRD,
                level: LevelMatch::HeightAboveGroundMeters(10),
                units: "m/s",
            }),
            FieldSelector {
                field: CanonicalField::VWind,
                vertical: VerticalSelector::HeightAboveGroundMeters(10),
                ..
            } => Ok(Self {
                parameters: PARAMETER_VGRD,
                level: LevelMatch::HeightAboveGroundMeters(10),
                units: "m/s",
            }),
            FieldSelector {
                field: CanonicalField::WindSpeed,
                vertical: VerticalSelector::HeightAboveGroundMeters(10),
                ..
            } => Ok(Self {
                parameters: PARAMETER_WIND_SPEED,
                level: LevelMatch::HeightAboveGroundMeters(10),
                units: "m/s",
            }),
            FieldSelector {
                field: CanonicalField::WindGust,
                vertical: VerticalSelector::HeightAboveGroundMeters(10),
                ..
            } => Ok(Self {
                parameters: PARAMETER_WIND_GUST,
                // Operational gust products are often keyed as 10 m AGL in
                // product catalogs even when the GRIB metadata carries a
                // surface level type.
                level: LevelMatch::SurfaceOrHeightAboveGroundMeters(10),
                units: "m/s",
            }),
            FieldSelector {
                field: CanonicalField::SmokeMassDensity,
                vertical: VerticalSelector::HeightAboveGroundMeters(8),
                ..
            } => Ok(Self {
                parameters: PARAMETER_SMOKE_MASS_DENSITY,
                level: LevelMatch::HeightAboveGroundMeters(8),
                units: "kg/m^3",
            }),
            FieldSelector {
                field: CanonicalField::PressureReducedToMeanSeaLevel,
                vertical: VerticalSelector::MeanSeaLevel,
                ..
            } => Ok(Self {
                parameters: PARAMETER_MSLP,
                level: LevelMatch::MeanSeaLevel,
                units: "Pa",
            }),
            FieldSelector {
                field: CanonicalField::PrecipitableWater,
                vertical: VerticalSelector::EntireAtmosphere,
                ..
            } => Ok(Self {
                parameters: PARAMETER_PWAT,
                level: LevelMatch::EntireAtmosphere,
                units: "kg/m^2",
            }),
            FieldSelector {
                field: CanonicalField::ColumnIntegratedSmoke,
                vertical: VerticalSelector::EntireAtmosphere,
                ..
            } => Ok(Self {
                parameters: PARAMETER_COLUMN_INTEGRATED_SMOKE,
                level: LevelMatch::EntireAtmosphere,
                units: "kg/m^2",
            }),
            FieldSelector {
                field: CanonicalField::TotalPrecipitation,
                vertical: VerticalSelector::Surface,
                ..
            } => Ok(Self {
                parameters: PARAMETER_TOTAL_PRECIPITATION,
                level: LevelMatch::Surface,
                units: "kg/m^2",
            }),
            FieldSelector {
                field: CanonicalField::ProbabilityOfPrecipitation,
                vertical: VerticalSelector::Surface,
                ..
            } => Ok(Self {
                parameters: PARAMETER_PROBABILITY_OF_PRECIPITATION,
                level: LevelMatch::Surface,
                units: "%",
            }),
            FieldSelector {
                field: CanonicalField::TotalCloudCover,
                vertical: VerticalSelector::EntireAtmosphere,
                ..
            } => Ok(Self {
                parameters: PARAMETER_TOTAL_CLOUD_COVER,
                level: LevelMatch::EntireAtmosphere,
                units: "%",
            }),
            FieldSelector {
                field: CanonicalField::LowCloudCover,
                vertical: VerticalSelector::EntireAtmosphere,
                ..
            } => Ok(Self {
                parameters: PARAMETER_LOW_CLOUD_COVER,
                level: LevelMatch::ExactLevelType(214),
                units: "%",
            }),
            FieldSelector {
                field: CanonicalField::MiddleCloudCover,
                vertical: VerticalSelector::EntireAtmosphere,
                ..
            } => Ok(Self {
                parameters: PARAMETER_MIDDLE_CLOUD_COVER,
                level: LevelMatch::ExactLevelType(224),
                units: "%",
            }),
            FieldSelector {
                field: CanonicalField::HighCloudCover,
                vertical: VerticalSelector::EntireAtmosphere,
                ..
            } => Ok(Self {
                parameters: PARAMETER_HIGH_CLOUD_COVER,
                level: LevelMatch::ExactLevelType(234),
                units: "%",
            }),
            FieldSelector {
                field: CanonicalField::Visibility,
                vertical: VerticalSelector::Surface,
                ..
            } => Ok(Self {
                parameters: PARAMETER_VISIBILITY,
                level: LevelMatch::Surface,
                units: "m",
            }),
            FieldSelector {
                field: CanonicalField::SimulatedInfraredBrightnessTemperature,
                vertical: VerticalSelector::NominalTop,
                ..
            } => Ok(Self {
                parameters: PARAMETER_SIMULATED_IR,
                level: LevelMatch::NominalTop,
                units: "K",
            }),
            FieldSelector {
                field: CanonicalField::CategoricalRain,
                vertical: VerticalSelector::Surface,
                ..
            } => Ok(Self {
                parameters: PARAMETER_CATEGORICAL_RAIN,
                level: LevelMatch::Surface,
                units: "0/1",
            }),
            FieldSelector {
                field: CanonicalField::CategoricalFreezingRain,
                vertical: VerticalSelector::Surface,
                ..
            } => Ok(Self {
                parameters: PARAMETER_CATEGORICAL_FREEZING_RAIN,
                level: LevelMatch::Surface,
                units: "0/1",
            }),
            FieldSelector {
                field: CanonicalField::CategoricalIcePellets,
                vertical: VerticalSelector::Surface,
                ..
            } => Ok(Self {
                parameters: PARAMETER_CATEGORICAL_ICE_PELLETS,
                level: LevelMatch::Surface,
                units: "0/1",
            }),
            FieldSelector {
                field: CanonicalField::CategoricalSnow,
                vertical: VerticalSelector::Surface,
                ..
            } => Ok(Self {
                parameters: PARAMETER_CATEGORICAL_SNOW,
                level: LevelMatch::Surface,
                units: "0/1",
            }),
            FieldSelector {
                field: CanonicalField::RadarReflectivity,
                vertical: VerticalSelector::AltitudeMeters(500),
                ..
            } => Ok(Self {
                parameters: PARAMETER_RADAR_REFLECTIVITY,
                level: LevelMatch::AltitudeMeters(500),
                units: "dBZ",
            }),
            FieldSelector {
                field: CanonicalField::RadarReflectivity,
                vertical: VerticalSelector::HeightAboveGroundMeters(1000),
                ..
            } => Ok(Self {
                parameters: PARAMETER_RADAR_REFLECTIVITY,
                level: LevelMatch::HeightAboveGroundMeters(1000),
                units: "dBZ",
            }),
            FieldSelector {
                field: CanonicalField::LandSeaMask,
                vertical: VerticalSelector::Surface,
                ..
            } => Ok(Self {
                parameters: PARAMETER_LANDSEA_MASK,
                level: LevelMatch::Surface,
                units: "fraction",
            }),
            FieldSelector {
                field: CanonicalField::CompositeReflectivity,
                vertical: VerticalSelector::AltitudeMeters(500),
                ..
            } => Ok(Self {
                parameters: PARAMETER_COMPOSITE_REFLECTIVITY,
                level: LevelMatch::AltitudeMeters(500),
                units: "dBZ",
            }),
            FieldSelector {
                field: CanonicalField::CompositeReflectivity,
                vertical: VerticalSelector::EntireAtmosphere,
                ..
            } => Ok(Self {
                parameters: PARAMETER_COMPOSITE_REFLECTIVITY,
                level: LevelMatch::EntireAtmosphere,
                units: "dBZ",
            }),
            FieldSelector {
                field: CanonicalField::UpdraftHelicity,
                vertical:
                    VerticalSelector::HeightAboveGroundLayerMeters {
                        bottom_m: 2000,
                        top_m: 5000,
                    },
                ..
            } => Ok(Self {
                parameters: PARAMETER_UPDRAFT_HELICITY,
                // HRRR/RRFS native UH fields surface the top of the AGL layer
                // in GRIB metadata; the operational 2-5 km UH product is the
                // 5000 m entry.
                level: LevelMatch::HeightAboveGroundMeters(5000),
                units: "m^2/s^2",
            }),
            _ => Err(IoError::UnsupportedStructuredSelector { selector }),
        }
    }
}

/// Upper-air levels the structured extractor will select: every 25 hPa from
/// 100 to 1000 inclusive — the operational plot levels plus the dense
/// store-ingest grid. Levels a product file does not carry surface as
/// partial-extraction misses, not errors.
///
/// NOTE: rustwx-models has a same-named fn with intentionally narrower
/// semantics ({200,250,300,500,700,850}): this one is what extraction can
/// admit; that one is what recipe validation/UI exposes.
fn is_supported_upper_air_level(level_hpa: u16) -> bool {
    (100..=1000).contains(&level_hpa) && level_hpa % 25 == 0
}

impl LevelMatch {
    fn matches(self, message: &Grib2Message) -> bool {
        match self {
            Self::Surface => message.product.level_type == 1,
            Self::MeanSeaLevel => message.product.level_type == 101,
            Self::AltitudeMeters(height_m) => {
                message.product.level_type == 102
                    && (message.product.level_value - f64::from(height_m)).abs() < 0.25
            }
            Self::IsobaricHpa(level_hpa) => {
                message.product.level_type == 100
                    && (normalize_pressure_level_hpa(message.product.level_value)
                        - f64::from(level_hpa))
                    .abs()
                        < 0.25
            }
            Self::HybridLevel(level) => {
                message.product.level_type == 105
                    && (message.product.level_value - f64::from(level)).abs() < 0.25
            }
            Self::EntireAtmosphere => matches!(message.product.level_type, 10 | 200),
            Self::NominalTop => message.product.level_type == 8,
            Self::ExactLevelType(level_type) => message.product.level_type == level_type,
            Self::HeightAboveGroundMeters(level_m) => {
                matches!(message.product.level_type, 103 | 118)
                    && (message.product.level_value - f64::from(level_m)).abs() < 0.25
            }
            Self::SurfaceOrHeightAboveGroundMeters(level_m) => {
                message.product.level_type == 1
                    || (matches!(message.product.level_type, 103 | 118)
                        && (message.product.level_value - f64::from(level_m)).abs() < 0.25)
            }
        }
    }
}

/// Memo key for per-extraction-call coordinate caching.
///
/// The grid-side work in `build_selected_field` (`grid_latlon`, the lat/lon
/// `flip_rows` for scan-mode bit 0x40, and the per-row longitude
/// normalization/rotation) reads *only* `message.grid` — `nx`, `ny`, and
/// `scan_mode` are themselves `GridDefinition` fields. The parser does not
/// retain the raw section 3 bytes, so the key is instead composed from
/// **every** field of the parsed `GridDefinition` (f64s compared by bit
/// pattern). Because the key is a total snapshot of the only input, equal
/// keys are guaranteed to produce identical lat/lon arrays and identical
/// per-row value rotations; over-keying on fields a particular template
/// ignores can only cost a memo miss, never a wrong hit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GridMemoKey {
    template: u16,
    nx: u32,
    ny: u32,
    lat1: u64,
    lon1: u64,
    lat2: u64,
    lon2: u64,
    dx: u64,
    dy: u64,
    latin1: u64,
    latin2: u64,
    lov: u64,
    scan_mode: u8,
    lad: u64,
    projection_center_flag: u8,
    n_parallel: u32,
    south_pole_lat: u64,
    south_pole_lon: u64,
    rotation_angle: u64,
    satellite_lat: u64,
    satellite_lon: u64,
    xp: u64,
    yp: u64,
    altitude: u64,
    pl: Option<Vec<u32>>,
    is_reduced: bool,
    num_data_points: u32,
    shape_of_earth: u8,
    resolution_flags: u8,
}

impl GridMemoKey {
    fn from_grid(grid: &GridDefinition) -> Self {
        Self {
            template: grid.template,
            nx: grid.nx,
            ny: grid.ny,
            lat1: grid.lat1.to_bits(),
            lon1: grid.lon1.to_bits(),
            lat2: grid.lat2.to_bits(),
            lon2: grid.lon2.to_bits(),
            dx: grid.dx.to_bits(),
            dy: grid.dy.to_bits(),
            latin1: grid.latin1.to_bits(),
            latin2: grid.latin2.to_bits(),
            lov: grid.lov.to_bits(),
            scan_mode: grid.scan_mode,
            lad: grid.lad.to_bits(),
            projection_center_flag: grid.projection_center_flag,
            n_parallel: grid.n_parallel,
            south_pole_lat: grid.south_pole_lat.to_bits(),
            south_pole_lon: grid.south_pole_lon.to_bits(),
            rotation_angle: grid.rotation_angle.to_bits(),
            satellite_lat: grid.satellite_lat.to_bits(),
            satellite_lon: grid.satellite_lon.to_bits(),
            xp: grid.xp.to_bits(),
            yp: grid.yp.to_bits(),
            altitude: grid.altitude.to_bits(),
            pl: grid.pl.clone(),
            is_reduced: grid.is_reduced,
            num_data_points: grid.num_data_points,
            shape_of_earth: grid.shape_of_earth,
            resolution_flags: grid.resolution_flags,
        }
    }
}

/// Memoized grid-side result: the post-normalization coordinate grid exactly
/// as `build_selected_field` historically produced it, plus the per-row
/// rotate-left amounts so the matching values-side rotation can be replayed
/// for every field that shares the grid.
struct GridMemoEntry {
    grid: LatLonGrid,
    row_wraps: Vec<usize>,
}

/// Per-extraction-call grid memo: each distinct `GridDefinition` resolves to
/// one slot (coordinate grid + row wraps + projection). The slot indices are
/// the `grid_index` values a values-only extraction hands out, and the
/// `SelectedField2D` lane clones its per-field grid out of the same slots —
/// one implementation, two output shapes.
struct GridMemo {
    index: HashMap<GridMemoKey, usize>,
    slots: Vec<(GridMemoEntry, Option<GridProjection>)>,
}

impl GridMemo {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
            slots: Vec::new(),
        }
    }

    /// Resolve (building on first use) the slot for one message's grid.
    fn slot_index(
        &mut self,
        message: &Grib2Message,
        shape: GridShape,
        selector: FieldSelector,
    ) -> Result<usize, IoError> {
        match self.index.entry(GridMemoKey::from_grid(&message.grid)) {
            Entry::Occupied(slot) => Ok(*slot.get()),
            Entry::Vacant(slot) => {
                let entry = build_grid_memo_entry(&message.grid, shape, selector)?;
                let projection = grid_projection_from_grib2_grid(&message.grid);
                self.slots.push((entry, projection));
                Ok(*slot.insert(self.slots.len() - 1))
            }
        }
    }

    fn into_shared_grids(self) -> Vec<SharedExtractionGrid> {
        self.slots
            .into_iter()
            .map(|(entry, projection)| SharedExtractionGrid {
                grid: entry.grid,
                projection,
            })
            .collect()
    }
}

fn build_grid_memo_entry(
    grid_def: &GridDefinition,
    shape: GridShape,
    selector: FieldSelector,
) -> Result<GridMemoEntry, IoError> {
    let nx = shape.nx;
    let ny = shape.ny;
    let (mut lat, mut lon) = grid_latlon(grid_def);
    if lat.is_empty() || lon.is_empty() {
        return Err(IoError::MissingGridCoordinates { selector });
    }
    if grid_def.scan_mode & 0x40 != 0 {
        flip_rows(&mut lat, nx, ny);
        flip_rows(&mut lon, nx, ny);
    }
    let row_wraps = normalize_and_rotate_longitude_grid_rows(&mut lat, &mut lon, nx, ny);
    let grid = LatLonGrid::new(
        shape,
        lat.into_iter().map(|value| value as f32).collect(),
        lon.into_iter().map(|value| value as f32).collect(),
    )?;
    Ok(GridMemoEntry { grid, row_wraps })
}

/// Build one field's bare values, memoizing the (expensive) coordinate-grid
/// computation per distinct `GridDefinition` within one extraction call.
/// Values-side normalization (unpack, alternating-i scan, row flip, row
/// rotation) stays per-field and is exactly the sequence the
/// `SelectedField2D` lane applies; the coordinate grid is referenced by
/// slot index instead of being cloned out per field.
fn build_field_values(
    message: &Grib2Message,
    selector: FieldSelector,
    units: &str,
    grid_memo: &mut GridMemo,
) -> Result<ExtractedFieldValues, IoError> {
    let nx = message.grid.nx as usize;
    let ny = message.grid.ny as usize;
    let shape = GridShape::new(nx, ny)?;
    let grid_index = grid_memo.slot_index(message, shape, selector)?;
    let entry = &grid_memo.slots[grid_index].0;
    let mut values = unpack_message(message).map_err(|err| IoError::Grib(err.to_string()))?;
    normalize_alternating_i_scan_rows(&mut values, nx, ny, message.grid.scan_mode);
    if message.grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut values, nx, ny);
    }
    rotate_rows_left(&mut values, nx, &entry.row_wraps);

    let values: Vec<f32> = values.into_iter().map(|value| value as f32).collect();
    if values.len() != entry.grid.shape.len() {
        return Err(IoError::Core(
            rustwx_core::RustwxError::InvalidFieldDataLength {
                expected: entry.grid.shape.len(),
                actual: values.len(),
            },
        ));
    }
    Ok(ExtractedFieldValues {
        selector,
        units: units.to_string(),
        values,
        grid_index,
    })
}

/// Build one `SelectedField2D`: [`build_field_values`] plus a per-field
/// clone of the shared coordinate grid (the historical output shape).
fn build_selected_field(
    message: &Grib2Message,
    selector: FieldSelector,
    units: &str,
    grid_memo: &mut GridMemo,
) -> Result<SelectedField2D, IoError> {
    let field_values = build_field_values(message, selector, units, grid_memo)?;
    let (entry, projection) = &grid_memo.slots[field_values.grid_index];
    let mut field = SelectedField2D::new(selector, units, entry.grid.clone(), field_values.values)?;
    if let Some(projection) = projection.clone() {
        field = field.with_projection(projection);
    }
    Ok(field)
}

// GRIB2 Code Table 4.5 level type 100 (isobaric surface) always encodes the
// pressure value in pascals. Converting to hectopascals is a plain /100. The
// old heuristic "only divide when > 2000" collapsed stratospheric levels
// (e.g. 700 Pa = 7 hPa) onto tropospheric hectopascal numbers (e.g. 700 hPa),
// which made GFS and RRFS-A pick the wrong 700 mb RH message (flat brown).
fn normalize_pressure_level_hpa(level_value_pa: f64) -> f64 {
    level_value_pa / 100.0
}

fn is_supported_hrrr_smoke_hybrid_level(level: u16) -> bool {
    (1..=HRRR_WRFNAT_HYBRID_LEVEL_COUNT).contains(&level)
}

fn longitude_midpoint(west_deg: f64, east_deg: f64) -> f64 {
    let west = normalize_longitude(west_deg);
    let mut east = normalize_longitude(east_deg);
    if east < west {
        east += 360.0;
    }
    west + (east - west) / 2.0
}

fn normalize_longitude(lon: f64) -> f64 {
    if lon > 180.0 { lon - 360.0 } else { lon }
}

/// Grid-side half of the longitude normalization: normalize longitudes and
/// rotate each lat/lon row so longitudes stay monotone. Returns the per-row
/// rotate-left amount (0 = untouched) so `rotate_rows_left` can replay the
/// identical rotation on each field's values.
fn normalize_and_rotate_longitude_grid_rows(
    lat: &mut [f64],
    lon: &mut [f64],
    nx: usize,
    ny: usize,
) -> Vec<usize> {
    let mut row_wraps = vec![0usize; ny];
    if nx == 0 || ny == 0 {
        return row_wraps;
    }

    for (row, row_wrap) in row_wraps.iter_mut().enumerate() {
        let start = row * nx;
        let end = start + nx;
        let lat_row = &mut lat[start..end];
        let lon_row = &mut lon[start..end];

        for lon_value in lon_row.iter_mut() {
            *lon_value = normalize_longitude(*lon_value);
        }

        if let Some(wrap_idx) = first_longitude_wrap(lon_row) {
            lat_row.rotate_left(wrap_idx);
            lon_row.rotate_left(wrap_idx);
            *row_wrap = wrap_idx;
        }
    }
    row_wraps
}

/// Values-side replay of the per-row rotation computed by
/// `normalize_and_rotate_longitude_grid_rows`.
fn rotate_rows_left(values: &mut [f64], nx: usize, row_wraps: &[usize]) {
    for (row, &wrap_idx) in row_wraps.iter().enumerate() {
        if wrap_idx == 0 {
            continue;
        }
        let start = row * nx;
        values[start..start + nx].rotate_left(wrap_idx);
    }
}

fn normalize_alternating_i_scan_rows(values: &mut [f64], nx: usize, ny: usize, scan_mode: u8) {
    if nx == 0 || ny == 0 || values.len() != nx * ny {
        return;
    }
    if scan_mode & 0x20 != 0 {
        // Adjacent points consecutive in j are not represented by the row-major
        // canonical grid used downstream. No supported production model uses it.
        return;
    }

    let base_i_negative = scan_mode & 0x80 != 0;
    let alternating_i = scan_mode & 0x10 != 0;
    if !base_i_negative && !alternating_i {
        return;
    }

    for row in 0..ny {
        let row_i_negative = base_i_negative ^ (alternating_i && row % 2 == 1);
        if !row_i_negative {
            continue;
        }
        let start = row * nx;
        values[start..start + nx].reverse();
    }
}

fn first_longitude_wrap(lon_row: &[f64]) -> Option<usize> {
    lon_row
        .windows(2)
        .position(|pair| pair[1] < pair[0])
        .map(|idx| idx + 1)
}

#[cfg(test)]
mod tests;
