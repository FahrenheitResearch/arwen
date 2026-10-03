//! ODIM_H5 Cartesian `IMAGE` / `MAX` decoder.
//!
//! This is deliberately separate from [`crate::odim`]. Polar `PVOL` and
//! `SCAN` objects decode into `recast_radar_core::Volume`; a Cartesian maximum
//! product does not contain rays, tilts, or gate geometry and must not be
//! disguised as one. The returned [`OdimCartesianGrid`] types the ODIM
//! projection, grid geometry, source metadata and missing-value encoding of
//! the `MAX` dataset for a gridded map-layer consumer, and keeps everything
//! else the file holds: every dataset (the `VSP` and `HSP` side projections
//! beside the maximum) with every plane in its stored type and raw codes
//! ([`OdimCartesianDataset`], [`OdimCartesianPlane`]), and every attribute of
//! every group, verbatim ([`OdimCartesianGrid::attrs`]). Physical values are
//! computed on request ([`OdimCartesianGrid::values`],
//! [`OdimCartesianGrid::value_at`]).
//!
//! IMGW-PIB POLRAD writes the quantity/scaling attributes on
//! `/datasetN/what` (rather than `/datasetN/data1/what`) and stores the
//! 2-D maximum in `/datasetN/data1/data`. Files may also contain `VSP` and
//! `HSP` side projections; the typed view is the one `MAX` dataset.

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};
use recast_radar_core::bounded_read::DecodeBudget;
use recast_radar_core::model::{ArrayBuf, AttrValue};

use crate::h5::{H5Attr, H5File};
use crate::{OdimError, Result};

/// Groups below a dataset or plane (`how` subgroups, ...) are read this deep.
const MAX_GROUP_DEPTH: usize = 8;

/// Radius used by PROJ's named `+ellps=sphere` ellipsoid.
///
/// IMGW's `projdef` uses that exact spelling. Keeping the named-sphere radius
/// here lets consumers reproduce the projection without silently substituting
/// WGS84 or BowEcho's generic 6371-km display sphere.
pub const PROJ_SPHERE_RADIUS_M: f64 = 6_370_997.0;

/// Canonical physical quantity carried by an ODIM Cartesian image.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OdimCartesianQuantity {
    /// Reflectivity (`DBZH`, `DBZV`, `DBZ`, `TH`, `TV`), dBZ.
    Reflectivity,
    /// Differential reflectivity (`ZDR`, `ZDRU`, `UZDR`), dB.
    DifferentialReflectivity,
    /// Correlation coefficient (`RHOHV`, `RHOHVU`, `URHOHV`).
    CorrelationCoefficient,
    /// Differential phase (`PHIDP`, `PHIDPU`, `UPHIDP`), degrees.
    DifferentialPhase,
    /// Specific differential phase (`KDP`, `KDPU`), degrees per km.
    SpecificDifferentialPhase,
    /// Rain rate (`RATE`), mm/h.
    RainRate,
    /// Echo top height (`HGHT`), metres.
    EchoTopHeight,
    /// Any other quantity code, trimmed.
    Other(String),
}

impl OdimCartesianQuantity {
    /// Parse an ODIM quantity code while preserving unfamiliar codes.
    pub fn from_code(code: &str) -> Self {
        match code.trim().to_ascii_uppercase().as_str() {
            "DBZH" | "DBZV" | "DBZ" | "TH" | "TV" => Self::Reflectivity,
            "ZDR" | "ZDRU" | "UZDR" => Self::DifferentialReflectivity,
            "RHOHV" | "RHOHVU" | "URHOHV" => Self::CorrelationCoefficient,
            "PHIDP" | "PHIDPU" | "UPHIDP" => Self::DifferentialPhase,
            "KDP" | "KDPU" => Self::SpecificDifferentialPhase,
            "RATE" => Self::RainRate,
            "HGHT" => Self::EchoTopHeight,
            _ => Self::Other(code.trim().to_owned()),
        }
    }

    /// Canonical display/storage units for the quantity, when known.
    pub fn units(&self) -> Option<&'static str> {
        match self {
            Self::Reflectivity => Some("dBZ"),
            Self::DifferentialReflectivity => Some("dB"),
            Self::CorrelationCoefficient => Some("1"),
            Self::DifferentialPhase => Some("deg"),
            Self::SpecificDifferentialPhase => Some("deg/km"),
            Self::RainRate => Some("mm/h"),
            Self::EchoTopHeight => Some("m"),
            Self::Other(_) => None,
        }
    }
}

/// Site metadata carried by the IMAGE root groups.
#[derive(Clone, Debug, PartialEq)]
pub struct OdimCartesianSite {
    /// Best in-file site id. IMGW's `/how system` (for example `RAM`) wins;
    /// otherwise this falls back through `NOD`, `RAD`, and `WMO` in source.
    pub id: String,
    /// Original ODIM `/what source` string, retained for provenance.
    pub source: String,
    /// Radar latitude, degrees north.
    pub latitude_deg: f64,
    /// Radar longitude, degrees east.
    pub longitude_deg: f64,
    /// Radar height above mean sea level, metres, when the file states it.
    pub height_m: Option<f64>,
}

/// One geodetic corner from an ODIM Cartesian `where` group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdimGeoPoint {
    /// Latitude, degrees north.
    pub latitude_deg: f64,
    /// Longitude, degrees east.
    pub longitude_deg: f64,
}

/// The four named ODIM Cartesian grid corners.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdimCartesianCorners {
    /// Corner of the lower-left cell (`LL_lat`, `LL_lon`).
    pub lower_left: OdimGeoPoint,
    /// Corner of the lower-right cell (`LR_lat`, `LR_lon`).
    pub lower_right: OdimGeoPoint,
    /// Corner of the upper-left cell (`UL_lat`, `UL_lon`).
    pub upper_left: OdimGeoPoint,
    /// Corner of the upper-right cell (`UR_lat`, `UR_lon`).
    pub upper_right: OdimGeoPoint,
}

/// A projection shape the decoder can identify without external PROJ state.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum OdimCartesianProjection {
    /// Spherical azimuthal-equidistant projection centered on one radar.
    AzimuthalEquidistantSphere {
        /// Latitude of the projection centre, degrees north.
        center_latitude_deg: f64,
        /// Longitude of the projection centre, degrees east.
        center_longitude_deg: f64,
        /// Radius of the sphere, metres.
        radius_m: f64,
        /// The `projdef` PROJ string, as written.
        projdef: String,
    },
}

/// Cartesian image geometry. Values are stored row-major as `[y, x]`.
#[derive(Clone, Debug, PartialEq)]
pub struct OdimCartesianGeometry {
    /// Columns (`xsize`).
    pub width: usize,
    /// Rows (`ysize`).
    pub height: usize,
    /// Column spacing (`xscale`), metres.
    pub x_spacing_m: f64,
    /// Row spacing (`yscale`), metres.
    pub y_spacing_m: f64,
    /// Lowest height the product covers (`minheight`), metres, when stated.
    pub min_height_m: Option<f64>,
    /// Highest height the product covers (`maxheight`), metres, when stated.
    pub max_height_m: Option<f64>,
    /// Geodetic corners of the grid.
    pub corners: OdimCartesianCorners,
}

impl OdimCartesianGeometry {
    /// Projected offset of a cell center from the grid/projection center.
    ///
    /// ODIM IMAGE data follows image row order: row zero is the north/top
    /// edge, while columns run west-to-east. IMGW's MAX grids are centered on
    /// the radar and publish all four geodetic corners as a cross-check.
    pub fn cell_center_offset_m(&self, x: usize, y: usize) -> Option<(f64, f64)> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let east_m = (x as f64 + 0.5 - self.width as f64 * 0.5) * self.x_spacing_m;
        let north_m = (self.height as f64 * 0.5 - y as f64 - 0.5) * self.y_spacing_m;
        Some((east_m, north_m))
    }
}

/// Raw-to-physical encoding from the selected dataset's `what` group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OdimCartesianEncoding {
    /// ODIM `gain`: physical = raw × gain + offset.
    pub gain: f64,
    /// ODIM `offset`.
    pub offset: f64,
    /// ODIM `nodata`: the raw value for "not scanned", when stated.
    pub nodata: Option<f64>,
    /// ODIM `undetect`: the raw value for "scanned, no echo", when stated.
    pub undetect: Option<f64>,
}

/// One `dataM` or `qualityK` group of a Cartesian dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct OdimCartesianPlane {
    /// Group name (`data1`, `quality1`).
    pub name: String,
    /// Shape of its `data` dataset, row-major (`[ysize, xsize]`).
    pub dims: Vec<usize>,
    /// The stored values as stored (`u8`, `i8`, `u16`, `i16`, `i32`,
    /// `u32`, `i64`, `f32`, `f64`; `u64` planes widen to `f64`).
    pub raw: ArrayBuf,
    /// Every attribute of the group, verbatim: its own under their names,
    /// those of its members as `<member>.<name>` (`what.gain`,
    /// `data.CLASS`, `how.<subgroup>.<name>`), an enumerated plane's
    /// members as `data.enum_values` / `data.enum_names`, a `legend` as
    /// `legend.values` / `legend.names`.
    pub attrs: Vec<(Box<str>, AttrValue)>,
    /// Its `qualityK` groups, by `K`.
    pub quality: Vec<OdimCartesianPlane>,
}

/// One `datasetN` of a Cartesian `IMAGE`.
#[derive(Clone, Debug, PartialEq)]
pub struct OdimCartesianDataset {
    /// Group name (`dataset1`).
    pub name: String,
    /// Every attribute of the group and of its members but the planes,
    /// verbatim (`what.product`, `where.xsize`, `how.<name>`).
    pub attrs: Vec<(Box<str>, AttrValue)>,
    /// Its `dataM` groups, by `M`.
    pub planes: Vec<OdimCartesianPlane>,
    /// Its own `qualityK` groups, by `K`.
    pub quality: Vec<OdimCartesianPlane>,
}

impl OdimCartesianDataset {
    /// A `what` attribute of the dataset.
    pub fn what(&self, name: &str) -> Option<&AttrValue> {
        let key = format!("what.{name}");
        self.attrs
            .iter()
            .find(|(have, _)| **have == *key)
            .map(|(_, value)| value)
    }
}

/// One decoded Cartesian maximum product.
#[derive(Clone, Debug, PartialEq)]
pub struct OdimCartesianGrid {
    /// ODIM root version string (IMGW currently writes `H5rd 2.3`).
    pub odim_version: Option<String>,
    /// Selected dataset group (normally `dataset1`).
    pub dataset: String,
    /// Product code; this decoder only returns `MAX`.
    pub product: String,
    /// Original ODIM quantity spelling.
    pub quantity_code: String,
    /// The quantity, classified.
    pub quantity: OdimCartesianQuantity,
    /// Units of the physical values, when known.
    pub units: Option<String>,
    /// Start of the product's time window (`startdate`, `starttime`).
    pub start_time: DateTime<Utc>,
    /// End of the product's time window, when stated.
    pub end_time: Option<DateTime<Utc>>,
    /// The radar the product belongs to.
    pub site: OdimCartesianSite,
    /// The grid's map projection.
    pub projection: OdimCartesianProjection,
    /// Size, spacing and corners of the grid.
    pub geometry: OdimCartesianGeometry,
    /// How raw values map to physical values.
    pub encoding: OdimCartesianEncoding,
    /// Every root attribute, verbatim: the root group's own
    /// (`Conventions`) and those of `/what`, `/where`, `/how` and any other
    /// root member but the datasets, as `<group>.<name>` (`what.object`,
    /// `how.system`, `how.<subgroup>.<name>`). The typed fields above are
    /// read from these.
    pub attrs: Vec<(Box<str>, AttrValue)>,
    /// Every `datasetN` of the file, by `N`: the `MAX` dataset
    /// ([`Self::dataset`]) and any others (IMGW's `VSP` and `HSP` side
    /// projections), each plane in its stored type.
    pub datasets: Vec<OdimCartesianDataset>,
}

impl OdimCartesianGrid {
    /// The `MAX` dataset's plane (its `data1`).
    pub fn max_plane(&self) -> Option<&OdimCartesianPlane> {
        self.datasets
            .iter()
            .find(|dataset| dataset.name == self.dataset)
            .and_then(|dataset| dataset.planes.first())
    }

    /// The stored codes of the maximum, row-major `[y, x]`.
    pub fn raw(&self) -> Option<&ArrayBuf> {
        self.max_plane().map(|plane| &plane.raw)
    }

    /// Physical value of cell `(x, y)` (row zero is north): `NaN` for the
    /// ODIM `nodata` and `undetect` codes; `None` outside the grid.
    pub fn value_at(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.geometry.width || y >= self.geometry.height {
            return None;
        }
        let raw = self.raw()?.get_f64(y * self.geometry.width + x)?;
        Some(physical(raw, self.encoding))
    }

    /// Row-major physical values of the maximum: `gain * raw + offset`,
    /// `NaN` for the `nodata` and `undetect` codes (which stay in
    /// [`Self::encoding`] and the raw plane).
    pub fn values(&self) -> Vec<f32> {
        let Some(raw) = self.raw() else {
            return Vec::new();
        };
        (0..raw.len())
            .map(|index| {
                raw.get_f64(index)
                    .map_or(f32::NAN, |raw| physical(raw, self.encoding))
            })
            .collect()
    }
}

/// `gain * raw + offset`, `NaN` for the sentinels and non-finite codes.
fn physical(raw: f64, encoding: OdimCartesianEncoding) -> f32 {
    if !raw.is_finite() || encoding.nodata == Some(raw) || encoding.undetect == Some(raw) {
        f32::NAN
    } else {
        (raw * encoding.gain + encoding.offset) as f32
    }
}

/// Decode one ODIM_H5 Cartesian `IMAGE` containing exactly one `MAX`
/// dataset.
///
/// This function is intentionally not part of
/// `recast_radar_io::read_supported_volume_bytes`: an IMAGE is a georeferenced
/// grid, not a radar volume. Callers must opt into the grid path explicitly.
pub fn decode_odim_h5_cartesian_max(bytes: &[u8]) -> Result<OdimCartesianGrid> {
    let file = H5File::open(bytes)?;
    let object = required_string(&file, "/what", "object")?;
    if object != "IMAGE" {
        return Err(invalid(format!(
            "ODIM_H5 object '{object}' is not a Cartesian IMAGE"
        )));
    }

    let dataset = select_max_dataset(&file)?;
    let what_path = format!("/{dataset}/what");
    let where_path = format!("/{dataset}/where");
    let product = required_string(&file, &what_path, "product")?;
    debug_assert_eq!(product, "MAX");
    let quantity_code = required_string(&file, &what_path, "quantity")?;
    let quantity = OdimCartesianQuantity::from_code(&quantity_code);
    let units = quantity.units().map(str::to_owned);

    let start_time = dataset_or_root_start_time(&file, &what_path)?;
    let end_time = optional_datetime(&file, &what_path, "enddate", "endtime")?;
    let site = decode_site(&file)?;
    let projection = decode_projection(&file, &where_path)?;
    let geometry = decode_geometry(&file, &where_path)?;
    let encoding = OdimCartesianEncoding {
        gain: optional_f64(&file, &what_path, "gain")?.unwrap_or(1.0),
        offset: optional_f64(&file, &what_path, "offset")?.unwrap_or(0.0),
        nodata: optional_f64(&file, &what_path, "nodata")?,
        undetect: optional_f64(&file, &what_path, "undetect")?,
    };

    let mut budget = DecodeBudget::volume();
    let datasets = read_datasets(&file, &mut budget)?;
    let plane = datasets
        .iter()
        .find(|candidate| candidate.name == dataset)
        .and_then(|candidate| candidate.planes.first())
        .ok_or_else(|| invalid(format!("ODIM_H5 {dataset} has no data1 plane")))?;
    let expected_dims = [geometry.height, geometry.width];
    if plane.dims.as_slice() != expected_dims {
        return Err(invalid(format!(
            "ODIM_H5 {dataset} MAX data shape {:?} does not match where ysize/xsize {:?}",
            plane.dims, expected_dims
        )));
    }
    let expected_len = geometry
        .width
        .checked_mul(geometry.height)
        .ok_or_else(|| invalid("ODIM_H5 Cartesian grid dimensions overflow"))?;
    if plane.raw.len() != expected_len {
        return Err(invalid(format!(
            "ODIM_H5 {dataset} MAX has {} values, expected {expected_len}",
            plane.raw.len()
        )));
    }
    let attrs = tree_attrs(
        &file,
        "",
        &|child| is_numbered(child, "dataset"),
        &mut budget,
    )?;

    Ok(OdimCartesianGrid {
        odim_version: optional_string(&file, "/what", "version"),
        dataset,
        product,
        quantity_code,
        quantity,
        units,
        start_time,
        end_time,
        site,
        projection,
        geometry,
        encoding,
        attrs,
        datasets,
    })
}

fn is_numbered(name: &str, prefix: &str) -> bool {
    name.strip_prefix(prefix)
        .is_some_and(|rest| rest.parse::<u32>().is_ok())
}

/// The members of `path` named `<prefix>N`, by `N`.
fn numbered_children(file: &H5File<'_>, path: &str, prefix: &str) -> Vec<String> {
    let mut names: Vec<String> = file
        .child_names(path)
        .into_iter()
        .filter(|name| is_numbered(name, prefix))
        .collect();
    names.sort_by_key(|name| name[prefix.len()..].parse::<u32>().unwrap_or(u32::MAX));
    names
}

/// Every attribute of `path` (bare names) and of its members but those
/// `skip` names (as `<member>.<name>`, recursively), charged to `budget`.
fn tree_attrs(
    file: &H5File<'_>,
    path: &str,
    skip: &dyn Fn(&str) -> bool,
    budget: &mut DecodeBudget,
) -> Result<Vec<(Box<str>, AttrValue)>> {
    let mut out = Vec::new();
    collect_attrs(file, path, "", skip, 0, &mut out);
    let bytes: usize = out
        .iter()
        .map(|(name, value)| name.len() + attr_bytes(value))
        .sum();
    budget
        .charge(bytes, 1, "ODIM_H5 Cartesian attributes")
        .map_err(OdimError::LimitExceeded)?;
    Ok(out)
}

fn collect_attrs(
    file: &H5File<'_>,
    path: &str,
    prefix: &str,
    skip: &dyn Fn(&str) -> bool,
    depth: usize,
    out: &mut Vec<(Box<str>, AttrValue)>,
) {
    if depth > MAX_GROUP_DEPTH {
        return;
    }
    let here = if path.is_empty() { "/" } else { path };
    for (name, value) in file.attr_entries(here) {
        out.push((format!("{prefix}{name}").into(), value));
    }
    for child in file.child_names(here) {
        if skip(&child) {
            continue;
        }
        let child_path = format!("{path}/{child}");
        if child == "legend"
            && let Some(legend) = file.legend(&child_path)
        {
            let (values, names): (Vec<i64>, Vec<Box<str>>) = legend
                .into_iter()
                .map(|(value, name)| (value, name.into()))
                .unzip();
            out.push((
                format!("{prefix}legend.values").into(),
                AttrValue::Array(ArrayBuf::I64(values)),
            ));
            out.push((
                format!("{prefix}legend.names").into(),
                AttrValue::Array(ArrayBuf::Text(names)),
            ));
        }
        collect_attrs(
            file,
            &child_path,
            &format!("{prefix}{child}."),
            &|_| false,
            depth + 1,
            out,
        );
    }
}

/// Approximate heap bytes of an attribute value.
fn attr_bytes(value: &AttrValue) -> usize {
    match value {
        AttrValue::Text(text) => text.len(),
        AttrValue::Array(ArrayBuf::Text(texts)) => texts.iter().map(|text| text.len() + 16).sum(),
        AttrValue::Array(array) => array.len() * 8,
        _ => 8,
    }
}

/// Every `datasetN` with its planes and quality groups.
fn read_datasets(
    file: &H5File<'_>,
    budget: &mut DecodeBudget,
) -> Result<Vec<OdimCartesianDataset>> {
    let mut out = Vec::new();
    for name in numbered_children(file, "/", "dataset") {
        let path = format!("/{name}");
        let attrs = tree_attrs(
            file,
            &path,
            &|child| is_numbered(child, "data") || is_numbered(child, "quality"),
            budget,
        )?;
        let mut planes = Vec::new();
        for plane in numbered_children(file, &path, "data") {
            planes.push(read_plane(
                file,
                &format!("{path}/{plane}"),
                &plane,
                budget,
            )?);
        }
        let mut quality = Vec::new();
        for group in numbered_children(file, &path, "quality") {
            quality.push(read_plane(
                file,
                &format!("{path}/{group}"),
                &group,
                budget,
            )?);
        }
        out.push(OdimCartesianDataset {
            name,
            attrs,
            planes,
            quality,
        });
    }
    Ok(out)
}

/// A `dataM` or `qualityK` group: its `data` dataset as stored, every
/// attribute, its quality groups.
fn read_plane(
    file: &H5File<'_>,
    path: &str,
    name: &str,
    budget: &mut DecodeBudget,
) -> Result<OdimCartesianPlane> {
    let (dims, raw, enum_members) = if file.has_object(&format!("{path}/data")) {
        file.dataset_raw(&format!("{path}/data"))?
    } else {
        (Vec::new(), ArrayBuf::U8(Vec::new()), Vec::new())
    };
    let word = match &raw {
        ArrayBuf::I8(_) | ArrayBuf::U8(_) => 1,
        ArrayBuf::I16(_) | ArrayBuf::U16(_) => 2,
        ArrayBuf::I32(_) | ArrayBuf::U32(_) | ArrayBuf::F32(_) => 4,
        _ => 8,
    };
    budget
        .charge(raw.len(), word, "ODIM_H5 Cartesian plane")
        .map_err(OdimError::LimitExceeded)?;
    let mut attrs = tree_attrs(file, path, &|child| is_numbered(child, "quality"), budget)?;
    if !enum_members.is_empty() {
        let (values, names): (Vec<i64>, Vec<Box<str>>) = enum_members
            .into_iter()
            .map(|(value, name)| (value, name.into()))
            .unzip();
        attrs.push((
            "data.enum_values".into(),
            AttrValue::Array(ArrayBuf::I64(values)),
        ));
        attrs.push((
            "data.enum_names".into(),
            AttrValue::Array(ArrayBuf::Text(names)),
        ));
    }
    let mut quality = Vec::new();
    for group in numbered_children(file, path, "quality") {
        quality.push(read_plane(
            file,
            &format!("{path}/{group}"),
            &group,
            budget,
        )?);
    }
    Ok(OdimCartesianPlane {
        name: name.to_owned(),
        dims,
        raw,
        attrs,
        quality,
    })
}

fn select_max_dataset(file: &H5File<'_>) -> Result<String> {
    let mut datasets: Vec<String> = file
        .child_names("/")
        .into_iter()
        .filter(|name| {
            name.strip_prefix("dataset")
                .is_some_and(|rest| rest.parse::<u32>().is_ok())
        })
        .collect();
    datasets.sort_by_key(|name| name[7..].parse::<u32>().unwrap_or(u32::MAX));
    let max_datasets: Vec<String> = datasets
        .into_iter()
        .filter(|name| {
            optional_string(file, &format!("/{name}/what"), "product").as_deref() == Some("MAX")
        })
        .collect();
    match max_datasets.as_slice() {
        [dataset] => Ok(dataset.clone()),
        [] => Err(invalid("ODIM_H5 IMAGE has no dataset with product=MAX")),
        many => Err(invalid(format!(
            "ODIM_H5 IMAGE has {} MAX datasets; expected exactly one",
            many.len()
        ))),
    }
}

fn decode_site(file: &H5File<'_>) -> Result<OdimCartesianSite> {
    let source = required_string(file, "/what", "source")?;
    let system = optional_string(file, "/how", "system")
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.trim().to_ascii_uppercase());
    let id = system.unwrap_or_else(|| site_id_from_source(&source));
    let latitude_deg = required_f64(file, "/where", "lat")?;
    let longitude_deg = required_f64(file, "/where", "lon")?;
    if !(-90.0..=90.0).contains(&latitude_deg) {
        return Err(invalid(format!(
            "ODIM_H5 IMAGE site latitude {latitude_deg} is outside [-90, 90]"
        )));
    }
    if !(-180.0..=180.0).contains(&longitude_deg) {
        return Err(invalid(format!(
            "ODIM_H5 IMAGE site longitude {longitude_deg} is outside [-180, 180]"
        )));
    }
    Ok(OdimCartesianSite {
        id,
        source,
        latitude_deg,
        longitude_deg,
        height_m: optional_f64(file, "/where", "height")?,
    })
}

fn site_id_from_source(source: &str) -> String {
    for wanted in ["NOD", "RAD", "WMO"] {
        if let Some(value) = source.split(',').find_map(|pair| {
            let (key, value) = pair.split_once(':')?;
            (key.trim() == wanted && !value.trim().is_empty()).then(|| value.trim())
        }) {
            return value.to_ascii_uppercase();
        }
    }
    "ODIM".to_owned()
}

fn decode_projection(file: &H5File<'_>, where_path: &str) -> Result<OdimCartesianProjection> {
    let projdef = required_string(file, where_path, "projdef")?;
    parse_projection(projdef)
}

fn parse_projection(projdef: String) -> Result<OdimCartesianProjection> {
    let proj = projdef_token(&projdef, "+proj=")
        .ok_or_else(|| invalid(format!("ODIM_H5 IMAGE projection has no +proj: {projdef}")))?;
    if proj != "aeqd" {
        return Err(invalid(format!(
            "ODIM_H5 IMAGE projection '+proj={proj}' unsupported; only spherical aeqd is supported"
        )));
    }
    let ellps = projdef_token(&projdef, "+ellps=").ok_or_else(|| {
        invalid(format!(
            "ODIM_H5 IMAGE aeqd projection has no +ellps: {projdef}"
        ))
    })?;
    if ellps != "sphere" {
        return Err(invalid(format!(
            "ODIM_H5 IMAGE aeqd ellipsoid '{ellps}' unsupported; expected +ellps=sphere"
        )));
    }
    if let Some(units) = projdef_token(&projdef, "+units=")
        && units != "m"
    {
        return Err(invalid(format!(
            "ODIM_H5 IMAGE aeqd units '{units}' unsupported; expected metres"
        )));
    }
    for key in ["+x_0=", "+y_0="] {
        if let Some(value) = projdef_f64(&projdef, key)?
            && value.abs() > f64::EPSILON
        {
            return Err(invalid(format!(
                "ODIM_H5 IMAGE aeqd non-zero {key}{value} unsupported"
            )));
        }
    }
    let center_latitude_deg = required_projdef_f64(&projdef, "+lat_0=")?;
    let center_longitude_deg = required_projdef_f64(&projdef, "+lon_0=")?;
    if !(-90.0..=90.0).contains(&center_latitude_deg)
        || !(-180.0..=180.0).contains(&center_longitude_deg)
    {
        return Err(invalid(format!(
            "ODIM_H5 IMAGE aeqd center is invalid: lat_0={center_latitude_deg}, lon_0={center_longitude_deg}"
        )));
    }
    Ok(OdimCartesianProjection::AzimuthalEquidistantSphere {
        center_latitude_deg,
        center_longitude_deg,
        radius_m: PROJ_SPHERE_RADIUS_M,
        projdef,
    })
}

fn decode_geometry(file: &H5File<'_>, where_path: &str) -> Result<OdimCartesianGeometry> {
    let width = required_usize(file, where_path, "xsize")?;
    let height = required_usize(file, where_path, "ysize")?;
    if width == 0 || height == 0 {
        return Err(invalid(
            "ODIM_H5 Cartesian grid dimensions must be non-zero",
        ));
    }
    let x_spacing_m = required_positive_f64(file, where_path, "xscale")?;
    let y_spacing_m = required_positive_f64(file, where_path, "yscale")?;
    Ok(OdimCartesianGeometry {
        width,
        height,
        x_spacing_m,
        y_spacing_m,
        min_height_m: optional_f64(file, where_path, "minheight")?,
        max_height_m: optional_f64(file, where_path, "maxheight")?,
        corners: OdimCartesianCorners {
            lower_left: required_corner(file, where_path, "LL")?,
            lower_right: required_corner(file, where_path, "LR")?,
            upper_left: required_corner(file, where_path, "UL")?,
            upper_right: required_corner(file, where_path, "UR")?,
        },
    })
}

fn required_corner(file: &H5File<'_>, where_path: &str, prefix: &str) -> Result<OdimGeoPoint> {
    let latitude_deg = required_f64(file, where_path, &format!("{prefix}_lat"))?;
    let longitude_deg = required_f64(file, where_path, &format!("{prefix}_lon"))?;
    if !(-90.0..=90.0).contains(&latitude_deg) || !(-180.0..=180.0).contains(&longitude_deg) {
        return Err(invalid(format!(
            "ODIM_H5 IMAGE {prefix} corner is invalid: lat={latitude_deg}, lon={longitude_deg}"
        )));
    }
    Ok(OdimGeoPoint {
        latitude_deg,
        longitude_deg,
    })
}

fn dataset_or_root_start_time(file: &H5File<'_>, what_path: &str) -> Result<DateTime<Utc>> {
    match optional_datetime(file, what_path, "startdate", "starttime")? {
        Some(value) => Ok(value),
        None => required_datetime(file, "/what", "date", "time"),
    }
}

fn optional_datetime(
    file: &H5File<'_>,
    path: &str,
    date_attr: &str,
    time_attr: &str,
) -> Result<Option<DateTime<Utc>>> {
    let date = optional_string(file, path, date_attr);
    let time = optional_string(file, path, time_attr);
    match (date, time) {
        (None, None) => Ok(None),
        (Some(date), Some(time)) => {
            parse_datetime(path, date_attr, time_attr, &date, &time).map(Some)
        }
        _ => Err(invalid(format!(
            "ODIM_H5 {path} must carry both {date_attr} and {time_attr}"
        ))),
    }
}

fn required_datetime(
    file: &H5File<'_>,
    path: &str,
    date_attr: &str,
    time_attr: &str,
) -> Result<DateTime<Utc>> {
    let date = required_string(file, path, date_attr)?;
    let time = required_string(file, path, time_attr)?;
    parse_datetime(path, date_attr, time_attr, &date, &time)
}

fn parse_datetime(
    path: &str,
    date_attr: &str,
    time_attr: &str,
    date: &str,
    time: &str,
) -> Result<DateTime<Utc>> {
    let date = NaiveDate::parse_from_str(date, "%Y%m%d")
        .map_err(|err| invalid(format!("ODIM_H5 {path}/{date_attr} is not YYYYMMDD: {err}")))?;
    let time = NaiveTime::parse_from_str(time, "%H%M%S")
        .map_err(|err| invalid(format!("ODIM_H5 {path}/{time_attr} is not HHMMSS: {err}")))?;
    Ok(Utc.from_utc_datetime(&NaiveDateTime::new(date, time)))
}

fn optional_string(file: &H5File<'_>, path: &str, name: &str) -> Option<String> {
    file.attr(path, name)
        .and_then(|attr| attr.as_str().map(str::to_owned))
}

fn required_string(file: &H5File<'_>, path: &str, name: &str) -> Result<String> {
    optional_string(file, path, name)
        .ok_or_else(|| invalid(format!("ODIM_H5 {path} has no string attribute '{name}'")))
}

fn optional_f64(file: &H5File<'_>, path: &str, name: &str) -> Result<Option<f64>> {
    let Some(attr) = file.attr(path, name) else {
        return Ok(None);
    };
    let value = attr
        .as_f64()
        .ok_or_else(|| invalid(format!("ODIM_H5 {path} attribute '{name}' is not numeric")))?;
    if !value.is_finite() {
        return Err(invalid(format!(
            "ODIM_H5 {path} attribute '{name}' is not finite"
        )));
    }
    Ok(Some(value))
}

fn required_f64(file: &H5File<'_>, path: &str, name: &str) -> Result<f64> {
    optional_f64(file, path, name)?
        .ok_or_else(|| invalid(format!("ODIM_H5 {path} has no numeric attribute '{name}'")))
}

fn required_positive_f64(file: &H5File<'_>, path: &str, name: &str) -> Result<f64> {
    let value = required_f64(file, path, name)?;
    if value <= 0.0 {
        return Err(invalid(format!(
            "ODIM_H5 {path} attribute '{name}' must be positive, got {value}"
        )));
    }
    Ok(value)
}

fn required_usize(file: &H5File<'_>, path: &str, name: &str) -> Result<usize> {
    let attr = file
        .attr(path, name)
        .ok_or_else(|| invalid(format!("ODIM_H5 {path} has no integer attribute '{name}'")))?;
    let value = match attr {
        H5Attr::I64(value) => value,
        H5Attr::F64(value) if value.is_finite() && value.fract() == 0.0 => value as i64,
        _ => {
            return Err(invalid(format!(
                "ODIM_H5 {path} attribute '{name}' is not an integer"
            )));
        }
    };
    usize::try_from(value).map_err(|_| {
        invalid(format!(
            "ODIM_H5 {path} attribute '{name}' is negative or too large: {value}"
        ))
    })
}

fn projdef_token<'a>(projdef: &'a str, prefix: &str) -> Option<&'a str> {
    projdef
        .split_whitespace()
        .find_map(|token| token.strip_prefix(prefix))
}

fn projdef_f64(projdef: &str, key: &str) -> Result<Option<f64>> {
    let Some(raw) = projdef_token(projdef, key) else {
        return Ok(None);
    };
    let value = raw.parse::<f64>().map_err(|err| {
        invalid(format!(
            "ODIM_H5 IMAGE projection token '{key}{raw}' is not numeric: {err}"
        ))
    })?;
    if !value.is_finite() {
        return Err(invalid(format!(
            "ODIM_H5 IMAGE projection token '{key}{raw}' is not finite"
        )));
    }
    Ok(Some(value))
}

fn required_projdef_f64(projdef: &str, key: &str) -> Result<f64> {
    projdef_f64(projdef, key)?.ok_or_else(|| {
        invalid(format!(
            "ODIM_H5 IMAGE projection has no {key} token: {projdef}"
        ))
    })
}

fn invalid(reason: impl Into<String>) -> OdimError {
    OdimError::InvalidMessage {
        offset: 0,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantity_codes_have_physical_units_and_unknowns_survive() {
        assert_eq!(
            OdimCartesianQuantity::from_code("KDP"),
            OdimCartesianQuantity::SpecificDifferentialPhase
        );
        assert_eq!(OdimCartesianQuantity::from_code("RhoHV").units(), Some("1"));
        assert_eq!(
            OdimCartesianQuantity::from_code("made-up"),
            OdimCartesianQuantity::Other("made-up".to_owned())
        );
    }

    #[test]
    fn unsupported_projection_and_ellipsoid_errors_are_explicit() {
        let laea = parse_projection("+proj=laea +lat_0=50 +lon_0=20 +ellps=sphere".to_owned())
            .expect_err("LAEA must not masquerade as AEQD")
            .to_string();
        assert!(laea.contains("+proj=laea"), "{laea}");
        assert!(laea.contains("only spherical aeqd"), "{laea}");

        let wgs84 = parse_projection("+proj=aeqd +lat_0=50 +lon_0=20 +ellps=WGS84".to_owned())
            .expect_err("ellipsoidal AEQD needs a different inverse")
            .to_string();
        assert!(wgs84.contains("WGS84"), "{wgs84}");
        assert!(wgs84.contains("expected +ellps=sphere"), "{wgs84}");
    }

    #[test]
    fn image_rows_map_from_north_to_south() {
        let geometry = OdimCartesianGeometry {
            width: 2,
            height: 2,
            x_spacing_m: 1000.0,
            y_spacing_m: 1000.0,
            min_height_m: None,
            max_height_m: None,
            corners: OdimCartesianCorners {
                lower_left: OdimGeoPoint {
                    latitude_deg: 0.0,
                    longitude_deg: 0.0,
                },
                lower_right: OdimGeoPoint {
                    latitude_deg: 0.0,
                    longitude_deg: 1.0,
                },
                upper_left: OdimGeoPoint {
                    latitude_deg: 1.0,
                    longitude_deg: 0.0,
                },
                upper_right: OdimGeoPoint {
                    latitude_deg: 1.0,
                    longitude_deg: 1.0,
                },
            },
        };
        assert_eq!(geometry.cell_center_offset_m(0, 0), Some((-500.0, 500.0)));
        assert_eq!(geometry.cell_center_offset_m(1, 1), Some((500.0, -500.0)));
        assert_eq!(geometry.cell_center_offset_m(2, 0), None);
    }
}
