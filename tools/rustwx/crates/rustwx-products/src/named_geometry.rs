use crate::places::{self, PlacePreset};
use crate::shared_context::DomainSpec;
use serde::{Deserialize, Serialize};
use shapefile::{Shape, dbase};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

const GROUP_COUNTRY: &str = "country";
const GROUP_GLOBAL_REGION: &str = "global_region";
const GROUP_US_REGION: &str = "us_region";
const GROUP_US_SPLIT_REGION: &str = "us_split_region";
const GROUP_US_MAJOR_METRO: &str = "us_major_metro";

const TAG_COUNTRY: &[&str] = &["country", "global", "admin0"];
const TAG_GLOBAL_REGION: &[&str] = &["global", "region"];
const TAG_US_REGION: &[&str] = &["us", "region"];
const TAG_US_SPLIT_REGION: &[&str] = &["us", "region", "split"];
const TAG_US_MAJOR_METRO: &[&str] = &["us", "metro", "major"];

const GROUPS_US_REGION: &[&str] = &[GROUP_US_REGION];
const GROUPS_GLOBAL_REGION: &[&str] = &[GROUP_GLOBAL_REGION];
const GROUPS_US_SPLIT_REGION: &[&str] = &[GROUP_US_SPLIT_REGION];
const GROUPS_US_REGION_AND_SPLIT: &[&str] = &[GROUP_US_REGION, GROUP_US_SPLIT_REGION];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NamedGeometryKind {
    Country,
    Metro,
    Region,
    WatchArea,
    Route,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NamedGeoPoint {
    pub lat_deg: f64,
    pub lon_deg: f64,
}

impl NamedGeoPoint {
    pub const fn new(lat_deg: f64, lon_deg: f64) -> Self {
        Self { lat_deg, lon_deg }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NamedGeoBounds {
    pub west_deg: f64,
    pub east_deg: f64,
    pub south_deg: f64,
    pub north_deg: f64,
}

impl NamedGeoBounds {
    pub const fn new(west_deg: f64, east_deg: f64, south_deg: f64, north_deg: f64) -> Self {
        Self {
            west_deg,
            east_deg,
            south_deg,
            north_deg,
        }
    }

    pub const fn as_tuple(self) -> (f64, f64, f64, f64) {
        (self.west_deg, self.east_deg, self.south_deg, self.north_deg)
    }

    pub fn center(self) -> NamedGeoPoint {
        let mut east = self.east_deg;
        if east < self.west_deg {
            east += 360.0;
        }
        NamedGeoPoint::new(
            (self.south_deg + self.north_deg) / 2.0,
            normalize_longitude((self.west_deg + east) / 2.0),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "geometry_type", rename_all = "snake_case")]
pub enum NamedGeometry {
    Bounds {
        bounds: NamedGeoBounds,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        center: Option<NamedGeoPoint>,
    },
    Path {
        points: Vec<NamedGeoPoint>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedGeometryAsset {
    pub slug: String,
    pub label: String,
    pub kind: NamedGeometryKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<String>,
    pub geometry: NamedGeometry,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl NamedGeometryAsset {
    pub fn bounds<S1: Into<String>, S2: Into<String>>(
        slug: S1,
        label: S2,
        kind: NamedGeometryKind,
        bounds: NamedGeoBounds,
    ) -> Self {
        Self {
            slug: slug.into(),
            label: label.into(),
            kind,
            groups: Vec::new(),
            geometry: NamedGeometry::Bounds {
                bounds,
                center: None,
            },
            tags: Vec::new(),
        }
    }

    pub fn route<S1: Into<String>, S2: Into<String>>(
        slug: S1,
        label: S2,
        points: Vec<NamedGeoPoint>,
    ) -> Self {
        Self {
            slug: slug.into(),
            label: label.into(),
            kind: NamedGeometryKind::Route,
            groups: Vec::new(),
            geometry: NamedGeometry::Path { points },
            tags: Vec::new(),
        }
    }

    pub fn with_center(mut self, center: NamedGeoPoint) -> Self {
        if let NamedGeometry::Bounds {
            center: existing, ..
        } = &mut self.geometry
        {
            *existing = Some(center);
        }
        self
    }

    pub fn with_group<S: Into<String>>(mut self, group: S) -> Self {
        push_unique_string(&mut self.groups, group.into());
        self
    }

    pub fn with_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags.clear();
        for tag in tags {
            push_unique_string(&mut self.tags, tag.into());
        }
        self
    }

    pub fn bounds_geometry(&self) -> Option<NamedGeoBounds> {
        match self.geometry {
            NamedGeometry::Bounds { bounds, .. } => Some(bounds),
            NamedGeometry::Path { .. } => None,
        }
    }

    pub fn path_points(&self) -> Option<&[NamedGeoPoint]> {
        match &self.geometry {
            NamedGeometry::Bounds { .. } => None,
            NamedGeometry::Path { points } => Some(points.as_slice()),
        }
    }

    pub fn domain_spec(&self) -> Option<DomainSpec> {
        self.bounds_geometry()
            .map(|bounds| DomainSpec::new(self.slug.clone(), bounds.as_tuple()))
    }

    pub fn has_group(&self, group: &str) -> bool {
        self.groups.iter().any(|candidate| candidate == group)
    }

    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|candidate| candidate == tag)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct NamedGeometrySelector {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<NamedGeometryKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slugs: Vec<String>,
}

impl NamedGeometrySelector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_kind(mut self, kind: NamedGeometryKind) -> Self {
        self.kind = Some(kind);
        self
    }

    pub fn with_group<S: Into<String>>(mut self, group: S) -> Self {
        self.group = Some(group.into());
        self
    }

    pub fn with_tag<S: Into<String>>(mut self, tag: S) -> Self {
        push_unique_string(&mut self.tags, tag.into());
        self
    }

    pub fn with_slug<S: Into<String>>(mut self, slug: S) -> Self {
        push_unique_string(&mut self.slugs, slug.into());
        self
    }

    fn matches(&self, asset: &NamedGeometryAsset) -> bool {
        if let Some(kind) = self.kind {
            if asset.kind != kind {
                return false;
            }
        }
        if let Some(group) = self.group.as_deref() {
            if !asset.has_group(group) {
                return false;
            }
        }
        if !self.slugs.is_empty() && !self.slugs.iter().any(|slug| slug == &asset.slug) {
            return false;
        }
        self.tags.iter().all(|tag| asset.has_tag(tag))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct NamedGeometryCatalog {
    pub assets: Vec<NamedGeometryAsset>,
}

impl NamedGeometryCatalog {
    pub fn new(assets: Vec<NamedGeometryAsset>) -> Self {
        Self { assets }
    }

    pub fn built_in() -> Self {
        built_in_named_geometry_catalog()
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    pub fn from_json_str(value: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(value)
    }

    pub fn load_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = fs::read(path)?;
        Ok(Self::from_json_slice(&bytes)?)
    }

    pub fn len(&self) -> usize {
        self.assets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &NamedGeometryAsset> {
        self.assets.iter()
    }

    pub fn find(&self, slug: &str) -> Option<&NamedGeometryAsset> {
        self.assets.iter().find(|asset| asset.slug == slug)
    }

    pub fn of_kind(&self, kind: NamedGeometryKind) -> Vec<&NamedGeometryAsset> {
        self.assets
            .iter()
            .filter(|asset| asset.kind == kind)
            .collect()
    }

    pub fn select<'a>(&'a self, selector: &NamedGeometrySelector) -> Vec<&'a NamedGeometryAsset> {
        self.assets
            .iter()
            .filter(|asset| selector.matches(asset))
            .collect()
    }

    pub fn domain_specs(&self, selector: &NamedGeometrySelector) -> Vec<DomainSpec> {
        self.select(selector)
            .into_iter()
            .filter_map(|asset| asset.domain_spec())
            .collect()
    }
}

pub fn built_in_named_geometry_catalog() -> NamedGeometryCatalog {
    let mut assets = Vec::<NamedGeometryAsset>::new();
    assets.extend(built_in_country_assets());
    assets.extend(built_in_region_assets());
    assets.extend(built_in_metro_assets());
    assets.extend(built_in_watch_area_assets());
    NamedGeometryCatalog::new(assets)
}

pub fn built_in_named_geometry_assets() -> Vec<NamedGeometryAsset> {
    built_in_named_geometry_catalog().assets
}

pub fn built_in_country_assets() -> Vec<NamedGeometryAsset> {
    static CACHE: OnceLock<Vec<NamedGeometryAsset>> = OnceLock::new();
    CACHE.get_or_init(load_country_assets).clone()
}

pub fn built_in_country_domains() -> Vec<DomainSpec> {
    built_in_country_assets()
        .into_iter()
        .filter_map(|asset| asset.domain_spec())
        .collect()
}

pub fn built_in_region_assets() -> Vec<NamedGeometryAsset> {
    BUILT_IN_ALL_REGION_PRESETS
        .iter()
        .copied()
        .map(BuiltInBoundsPreset::to_asset)
        .collect()
}

pub fn built_in_standard_region_assets() -> Vec<NamedGeometryAsset> {
    built_in_region_assets()
        .into_iter()
        .filter(|asset| asset.has_group(GROUP_US_REGION))
        .collect()
}

pub fn built_in_split_region_assets() -> Vec<NamedGeometryAsset> {
    built_in_region_assets()
        .into_iter()
        .filter(|asset| asset.has_group(GROUP_US_SPLIT_REGION))
        .collect()
}

pub fn built_in_region_domains() -> Vec<DomainSpec> {
    built_in_region_assets()
        .into_iter()
        .filter_map(|asset| asset.domain_spec())
        .collect()
}

pub fn built_in_standard_region_domains() -> Vec<DomainSpec> {
    built_in_standard_region_assets()
        .into_iter()
        .filter_map(|asset| asset.domain_spec())
        .collect()
}

pub fn built_in_split_region_domains() -> Vec<DomainSpec> {
    built_in_split_region_assets()
        .into_iter()
        .filter_map(|asset| asset.domain_spec())
        .collect()
}

pub fn built_in_metro_assets() -> Vec<NamedGeometryAsset> {
    places::major_us_city_places()
        .iter()
        .copied()
        .map(metro_asset_from_place)
        .collect()
}

pub fn built_in_watch_area_assets() -> Vec<NamedGeometryAsset> {
    Vec::new()
}

pub fn find_built_in_named_geometry(slug: &str) -> Option<NamedGeometryAsset> {
    built_in_named_geometry_catalog().find(slug).cloned()
}

pub fn find_built_in_country_asset(query: &str) -> Option<NamedGeometryAsset> {
    let key = country_lookup_key(query);
    if key.is_empty() {
        return None;
    }

    built_in_country_assets()
        .into_iter()
        .find(|asset| country_asset_matches(asset, &key))
}

pub fn find_built_in_country_domain(query: &str) -> Option<DomainSpec> {
    find_built_in_country_asset(query).and_then(|asset| asset.domain_spec())
}

fn load_country_assets() -> Vec<NamedGeometryAsset> {
    let Some(root) = rustwx_render::checked_in_natural_earth_110m_root() else {
        return Vec::new();
    };
    let path = root.join("ne_110m_admin_0_countries.shp");
    let Ok(mut reader) = shapefile::Reader::from_path(&path) else {
        return Vec::new();
    };

    let mut assets = Vec::<NamedGeometryAsset>::new();
    for item in reader.iter_shapes_and_records() {
        let Ok((shape, record)) = item else {
            continue;
        };
        let Some(asset) = country_asset_from_shape_record(&shape, &record) else {
            continue;
        };
        if !assets.iter().any(|existing| existing.slug == asset.slug) {
            assets.push(asset);
        }
    }
    assets.sort_by(|left, right| left.slug.cmp(&right.slug));
    assets
}

fn country_asset_from_shape_record(
    shape: &Shape,
    record: &dbase::Record,
) -> Option<NamedGeometryAsset> {
    let iso_a3 = record_text(record, &["ISO_A3", "ADM0_A3", "SOV_A3"])?;
    let iso_a2 = record_text(record, &["ISO_A2", "ISO_A2_EH"]);
    let label = record_text(record, &["NAME_LONG", "ADMIN", "NAME"])?;
    let continent = record_text(record, &["CONTINENT"]);
    let slug = iso_a3.to_ascii_lowercase();
    let points = lon_lat_points(shape);
    let bounds = country_bounds_from_points(&points)?;
    let center = bounds.center();

    let mut tags = TAG_COUNTRY
        .iter()
        .map(|tag| (*tag).to_string())
        .collect::<Vec<_>>();
    tags.push(format!("iso_a3:{}", iso_a3.to_ascii_lowercase()));
    if let Some(iso_a2) = iso_a2 {
        tags.push(format!("iso_a2:{}", iso_a2.to_ascii_lowercase()));
    }
    if let Some(continent) = continent {
        tags.push(format!("continent:{}", country_lookup_key(&continent)));
    }

    Some(
        NamedGeometryAsset::bounds(slug, label, NamedGeometryKind::Country, bounds)
            .with_center(center)
            .with_group(GROUP_COUNTRY)
            .with_tags(tags),
    )
}

fn record_text(record: &dbase::Record, fields: &[&str]) -> Option<String> {
    for field in fields {
        let Some(value) = record.get(field) else {
            continue;
        };
        if let dbase::FieldValue::Character(Some(text)) = value {
            let text = text.trim();
            if !text.is_empty() && text != "-99" {
                return Some(text.to_string());
            }
        }
    }
    None
}

fn lon_lat_points(shape: &Shape) -> Vec<(f64, f64)> {
    match shape {
        Shape::Polygon(polygon) => polygon
            .rings()
            .iter()
            .flat_map(|ring| ring.points().iter().map(|point| (point.x, point.y)))
            .filter(|(lon, lat)| lon.is_finite() && lat.is_finite())
            .collect(),
        _ => Vec::new(),
    }
}

fn country_bounds_from_points(points: &[(f64, f64)]) -> Option<NamedGeoBounds> {
    let mut lons = Vec::with_capacity(points.len());
    let mut south = f64::INFINITY;
    let mut north = f64::NEG_INFINITY;
    for &(lon, lat) in points {
        if !lon.is_finite() || !lat.is_finite() {
            continue;
        }
        lons.push(normalize_longitude(lon));
        south = south.min(lat);
        north = north.max(lat);
    }
    if lons.is_empty() || !south.is_finite() || !north.is_finite() {
        return None;
    }

    let (mut west, mut east) = minimal_longitude_bounds(&lons)?;
    let lon_span = longitude_span(west, east);
    let lat_span = (north - south).max(0.0);
    let lon_pad = ((lon_span * 0.06).max(0.60)).min(6.0);
    let lat_pad = ((lat_span * 0.06).max(0.60)).min(6.0);

    if lon_span + lon_pad * 2.0 >= 359.0 {
        west = -180.0;
        east = 180.0;
    } else {
        west = normalize_longitude(west - lon_pad);
        east = normalize_longitude(east + lon_pad);
    }
    south = (south - lat_pad).max(-89.5);
    north = (north + lat_pad).min(89.5);

    let min_span = 3.0;
    if longitude_span(west, east) < min_span {
        let center = longitude_midpoint(west, east);
        west = normalize_longitude(center - min_span / 2.0);
        east = normalize_longitude(center + min_span / 2.0);
    }
    if north - south < min_span {
        let center = ((north + south) / 2.0).clamp(-89.0, 89.0);
        south = (center - min_span / 2.0).max(-89.5);
        north = (center + min_span / 2.0).min(89.5);
    }

    Some(NamedGeoBounds::new(west, east, south, north))
}

fn minimal_longitude_bounds(lons: &[f64]) -> Option<(f64, f64)> {
    let mut sorted = lons
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(normalize_longitude)
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    sorted.dedup_by(|left, right| (*left - *right).abs() < 1.0e-9);
    if sorted.len() == 1 {
        return Some((sorted[0], sorted[0]));
    }

    let mut largest_gap = f64::NEG_INFINITY;
    let mut largest_gap_start = 0usize;
    for idx in 0..sorted.len() {
        let current = sorted[idx];
        let next = if idx + 1 < sorted.len() {
            sorted[idx + 1]
        } else {
            sorted[0] + 360.0
        };
        let gap = next - current;
        if gap > largest_gap {
            largest_gap = gap;
            largest_gap_start = idx;
        }
    }

    let west = sorted[(largest_gap_start + 1) % sorted.len()];
    let east = sorted[largest_gap_start];
    Some((normalize_longitude(west), normalize_longitude(east)))
}

fn country_asset_matches(asset: &NamedGeometryAsset, key: &str) -> bool {
    country_lookup_key(&asset.slug) == key
        || country_lookup_key(&asset.label) == key
        || asset.tags.iter().any(|tag| {
            let tail = tag.rsplit_once(':').map(|(_, value)| value).unwrap_or(tag);
            country_lookup_key(tail) == key
        })
}

fn country_lookup_key(value: &str) -> String {
    let mut key = String::with_capacity(value.len());
    let mut last_was_separator = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            key.push(ch.to_ascii_lowercase());
            last_was_separator = false;
        } else if !key.is_empty() && !last_was_separator {
            key.push('_');
            last_was_separator = true;
        }
    }
    key.trim_matches('_').to_string()
}

fn longitude_span(west: f64, east: f64) -> f64 {
    let west = normalize_longitude(west);
    let mut east = normalize_longitude(east);
    if east < west {
        east += 360.0;
    }
    (east - west).clamp(0.0, 360.0)
}

fn longitude_midpoint(west: f64, east: f64) -> f64 {
    let west = normalize_longitude(west);
    let mut east = normalize_longitude(east);
    if east < west {
        east += 360.0;
    }
    normalize_longitude((west + east) / 2.0)
}

fn normalize_longitude(lon: f64) -> f64 {
    let mut normalized = lon % 360.0;
    if normalized > 180.0 {
        normalized -= 360.0;
    } else if normalized <= -180.0 {
        normalized += 360.0;
    }
    normalized
}

fn metro_asset_from_place(place: PlacePreset) -> NamedGeometryAsset {
    NamedGeometryAsset::bounds(
        place.slug,
        place.label,
        NamedGeometryKind::Metro,
        NamedGeoBounds::from(place.bounds()),
    )
    .with_center(NamedGeoPoint::new(place.center_lat, place.center_lon))
    .with_group(GROUP_US_MAJOR_METRO)
    .with_tags(TAG_US_MAJOR_METRO.iter().copied())
}

fn push_unique_string(values: &mut Vec<String>, value: String) {
    if value.is_empty() || values.iter().any(|existing| existing == &value) {
        return;
    }
    values.push(value);
}

impl From<(f64, f64, f64, f64)> for NamedGeoBounds {
    fn from(value: (f64, f64, f64, f64)) -> Self {
        Self::new(value.0, value.1, value.2, value.3)
    }
}

#[derive(Debug, Clone, Copy)]
struct BuiltInBoundsPreset {
    slug: &'static str,
    label: &'static str,
    kind: NamedGeometryKind,
    groups: &'static [&'static str],
    tags: &'static [&'static str],
    bounds: NamedGeoBounds,
}

impl BuiltInBoundsPreset {
    fn to_asset(self) -> NamedGeometryAsset {
        let mut asset = NamedGeometryAsset::bounds(self.slug, self.label, self.kind, self.bounds)
            .with_tags(self.tags.iter().copied());
        for group in self.groups {
            asset = asset.with_group(*group);
        }
        asset
    }
}

const BUILT_IN_ALL_REGION_PRESETS: &[BuiltInBoundsPreset] = &[
    BuiltInBoundsPreset {
        slug: "global",
        label: "Global",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_GLOBAL_REGION,
        tags: TAG_GLOBAL_REGION,
        bounds: NamedGeoBounds::new(-180.0, 179.999, -90.0, 90.0),
    },
    BuiltInBoundsPreset {
        slug: "midwest",
        label: "Midwest",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_REGION,
        tags: TAG_US_REGION,
        bounds: NamedGeoBounds::new(-104.0, -74.0, 28.0, 49.0),
    },
    BuiltInBoundsPreset {
        slug: "conus",
        label: "CONUS",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_REGION,
        tags: TAG_US_REGION,
        bounds: NamedGeoBounds::new(-127.0, -66.0, 23.0, 51.5),
    },
    BuiltInBoundsPreset {
        slug: "california",
        label: "California",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_REGION,
        tags: TAG_US_REGION,
        bounds: NamedGeoBounds::new(-124.9, -113.8, 31.9, 42.5),
    },
    BuiltInBoundsPreset {
        slug: "california_square",
        label: "California Square",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_REGION,
        tags: TAG_US_REGION,
        bounds: NamedGeoBounds::new(-124.9, -113.7, 31.8, 42.7),
    },
    BuiltInBoundsPreset {
        slug: "reno_square",
        label: "Reno Square",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_REGION,
        tags: TAG_US_REGION,
        bounds: NamedGeoBounds::new(-123.1, -116.1, 36.1, 43.1),
    },
    BuiltInBoundsPreset {
        slug: "southeast",
        label: "Southeast",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_REGION_AND_SPLIT,
        tags: TAG_US_SPLIT_REGION,
        bounds: NamedGeoBounds::new(-96.0, -72.0, 24.0, 38.5),
    },
    BuiltInBoundsPreset {
        slug: "southern_plains",
        label: "Southern Plains",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_REGION_AND_SPLIT,
        tags: TAG_US_SPLIT_REGION,
        bounds: NamedGeoBounds::new(-109.0, -90.0, 25.0, 40.5),
    },
    BuiltInBoundsPreset {
        slug: "oklahoma",
        label: "Oklahoma",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_REGION_AND_SPLIT,
        tags: TAG_US_SPLIT_REGION,
        bounds: NamedGeoBounds::new(-103.75, -93.5, 32.75, 38.25),
    },
    BuiltInBoundsPreset {
        slug: "northeast",
        label: "Northeast",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_REGION_AND_SPLIT,
        tags: TAG_US_SPLIT_REGION,
        bounds: NamedGeoBounds::new(-84.5, -65.0, 36.0, 48.5),
    },
    BuiltInBoundsPreset {
        slug: "great_lakes",
        label: "Great Lakes",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_REGION_AND_SPLIT,
        tags: TAG_US_SPLIT_REGION,
        bounds: NamedGeoBounds::new(-97.5, -72.0, 39.0, 50.5),
    },
    BuiltInBoundsPreset {
        slug: "pacific_northwest",
        label: "Pacific Northwest",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_SPLIT_REGION,
        tags: TAG_US_SPLIT_REGION,
        bounds: NamedGeoBounds::new(-125.0, -110.0, 41.0, 49.5),
    },
    BuiltInBoundsPreset {
        slug: "california_southwest",
        label: "California / Southwest",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_SPLIT_REGION,
        tags: TAG_US_SPLIT_REGION,
        bounds: NamedGeoBounds::new(-125.0, -108.0, 31.0, 41.5),
    },
    BuiltInBoundsPreset {
        slug: "rockies_high_plains",
        label: "Rockies / High Plains",
        kind: NamedGeometryKind::Region,
        groups: GROUPS_US_SPLIT_REGION,
        tags: TAG_US_SPLIT_REGION,
        bounds: NamedGeoBounds::new(-112.0, -96.0, 37.0, 49.5),
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn built_in_catalog_spans_regions_and_metros() {
        let catalog = NamedGeometryCatalog::built_in();
        let kinds = catalog
            .iter()
            .map(|asset| asset.kind)
            .collect::<HashSet<_>>();

        assert!(kinds.contains(&NamedGeometryKind::Country));
        assert!(kinds.contains(&NamedGeometryKind::Region));
        assert!(kinds.contains(&NamedGeometryKind::Metro));
    }

    #[test]
    fn built_in_catalog_slugs_are_unique() {
        let catalog = NamedGeometryCatalog::built_in();
        let mut seen = HashSet::new();

        for asset in catalog.iter() {
            assert!(
                seen.insert(asset.slug.as_str()),
                "duplicate named geometry slug {}",
                asset.slug
            );
        }
    }

    #[test]
    fn selector_filters_by_kind_group_and_tag() {
        let catalog = NamedGeometryCatalog::built_in();
        let selector = NamedGeometrySelector::new()
            .with_kind(NamedGeometryKind::Region)
            .with_group(GROUP_US_SPLIT_REGION)
            .with_tag("split");
        let selected = catalog.select(&selector);

        assert!(!selected.is_empty());
        assert!(
            selected
                .iter()
                .all(|asset| asset.kind == NamedGeometryKind::Region)
        );
        assert!(
            selected
                .iter()
                .all(|asset| asset.has_group(GROUP_US_SPLIT_REGION))
        );
        assert!(selected.iter().all(|asset| asset.has_tag("split")));
    }

    #[test]
    fn country_assets_load_from_checked_in_natural_earth() {
        let countries = built_in_country_assets();
        assert!(
            countries.len() >= 150,
            "expected Natural Earth admin-0 countries, got {}",
            countries.len()
        );

        let usa = find_built_in_country_asset("us").expect("US lookup should resolve");
        assert_eq!(usa.slug, "usa");
        assert_eq!(usa.kind, NamedGeometryKind::Country);
        assert!(usa.has_group(GROUP_COUNTRY));
        assert!(usa.has_tag("iso_a3:usa"));
        assert!(usa.has_tag("iso_a2:us"));
        assert_eq!(
            find_built_in_country_domain("united_states").unwrap().slug,
            "usa"
        );
    }

    #[test]
    fn country_bounds_support_antimeridian_crops() {
        let fiji = find_built_in_country_domain("fji").expect("Fiji should resolve");
        assert!(
            fiji.bounds.0 > fiji.bounds.1,
            "Fiji should use wrapped west/east bounds, got {:?}",
            fiji.bounds
        );
    }

    #[test]
    fn global_region_is_available_as_a_named_domain() {
        let global = find_built_in_named_geometry("global").expect("global domain should resolve");
        assert_eq!(global.kind, NamedGeometryKind::Region);
        assert!(global.has_group(GROUP_GLOBAL_REGION));
        assert!(global.has_tag("global"));
        assert_eq!(
            global.domain_spec().unwrap().bounds,
            (-180.0, 179.999, -90.0, 90.0)
        );
    }

    #[test]
    fn domain_specs_skip_route_assets() {
        let catalog = NamedGeometryCatalog::new(vec![NamedGeometryAsset::route(
            "test_route",
            "Test Route",
            vec![
                NamedGeoPoint::new(35.2220, -101.8313),
                NamedGeoPoint::new(41.8781, -87.6298),
            ],
        )]);
        let selector = NamedGeometrySelector::new().with_slug("test_route");

        assert!(catalog.domain_specs(&selector).is_empty());
    }

    #[test]
    fn json_loader_supports_external_watch_area_catalogs() {
        let catalog = NamedGeometryCatalog::from_json_str(
            r#"{
                "assets": [
                    {
                        "slug": "foothill_watch",
                        "label": "Foothill Watch",
                        "kind": "watch_area",
                        "groups": ["enterprise_watch"],
                        "geometry": {
                            "geometry_type": "bounds",
                            "bounds": {
                                "west_deg": -122.5,
                                "east_deg": -121.5,
                                "south_deg": 38.1,
                                "north_deg": 39.0
                            }
                        },
                        "tags": ["enterprise", "fire"]
                    }
                ]
            }"#,
        )
        .expect("watch area catalog should deserialize");
        let asset = catalog
            .find("foothill_watch")
            .expect("watch area slug should be present");

        assert_eq!(asset.kind, NamedGeometryKind::WatchArea);
        assert!(asset.has_group("enterprise_watch"));
        assert_eq!(
            asset
                .domain_spec()
                .expect("watch area bounds should map to a domain"),
            DomainSpec::new("foothill_watch", (-122.5, -121.5, 38.1, 39.0))
        );
    }

    #[test]
    fn metro_assets_preserve_place_centers() {
        let preset = places::major_us_city_places()
            .iter()
            .find(|candidate| candidate.slug == "ca_los_angeles")
            .expect("Los Angeles metro preset should exist");
        let asset = built_in_metro_assets()
            .into_iter()
            .find(|candidate| candidate.slug == "ca_los_angeles")
            .expect("Los Angeles metro should exist");

        match asset.geometry {
            NamedGeometry::Bounds {
                center: Some(center),
                ..
            } => {
                assert!((center.lat_deg - preset.center_lat).abs() < 1.0e-6);
                assert!((center.lon_deg - preset.center_lon).abs() < 1.0e-6);
            }
            _ => panic!("metro geometry should carry a center point"),
        }
    }
}
