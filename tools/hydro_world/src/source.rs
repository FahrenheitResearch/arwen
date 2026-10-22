//! The builder's source table (`arwen.hydro-world.sources.v1`). A new region or
//! dataset is a new row, never a new code path: each row names the river
//! network, catchment geometry, DEM, soil, land cover, lakes/dams and gauge
//! feeds plus the attribute mapping, and the builder's stages read only the row.
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

pub const SCHEMA: &str = "arwen.hydro-world.sources.v1";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceTable {
    pub schema: String,
    /// `--row auto`: rules tried in order; the first whose limits the box
    /// meets names the row (finer DEM for small boxes, coarser otherwise).
    #[serde(default)]
    pub auto: Vec<AutoRule>,
    pub rows: Vec<SourceRow>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutoRule {
    pub row: String,
    /// The row applies when its working grid over the box has at most this
    /// many cells (absent: always).
    #[serde(default)]
    pub max_cells: Option<usize>,
    /// When the row's grid over the box has more than `max_cells` cells,
    /// keep the row and build on its grid coarsened by the smallest integer
    /// factor that fits (each cell the block mean of the row's own grid),
    /// instead of passing to the next rule. A last rule with this set means
    /// every box on Earth builds.
    #[serde(default)]
    pub coarsen: bool,
    /// Why this limit (memory and time per box, measured).
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRow {
    /// Row id, written to fabric_metadata.source_row.
    pub id: String,
    /// Lane that owns the row's recipe.
    pub owner: String,
    /// Fabric numeric id namespace: n = id_namespace * 10^10 + local id.
    pub id_namespace: i64,
    /// Written to network.hf_source.
    pub hf_source: String,
    /// Fabric CRS policy: "EPSG:<code>" or "laea-domain-centroid".
    pub crs: String,
    /// Vertical datum all elevations in the fabric use.
    pub vertical_datum: String,
    pub inputs: BTreeMap<String, Input>,
    /// model_attributes column -> derivation recipe id (rows in the parameter table).
    pub attribute_mapping: BTreeMap<String, String>,
    /// Gauge feeds whose stations become hydrolocations (hl_reference "Gages").
    pub gauges: Vec<GaugeFeed>,
    /// How the builder turns the inputs into a fabric (absent for rows whose
    /// recipe lives in another lane's tool, e.g. the CONUS NHDPlus row).
    #[serde(default)]
    pub build: Option<BuildSpec>,
}

/// Builder method and its parameters. Every number the builder uses comes
/// from here or from the recipe table, never from code.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildSpec {
    /// "dem-d8": network and catchments derived from the DEM (fill, D8,
    /// accumulation, streams); "vector-basins": river polylines with
    /// downstream ids plus basin polygons with downstream ids.
    pub method: String,
    /// dem-d8: working grid resolution in arc-seconds (the DEM's own spacing).
    #[serde(default)]
    pub grid_arcsec: Option<f64>,
    /// dem-d8: Priority-Flood tile edge in cells.
    #[serde(default)]
    pub fill_tile: Option<usize>,
    /// dem-d8: exact DEM value the dataset uses for sea (Copernicus: 0).
    #[serde(default)]
    pub sea_value: Option<f32>,
    /// dem-d8: upstream area at which a cell becomes a stream (km2).
    #[serde(default)]
    pub stream_threshold_km2: Option<f64>,
    /// dem-d8: reaches shorter than this (km) are merged into their downstream
    /// reach (NextGen-style refactoring); terminal and lake-outlet reaches are kept.
    #[serde(default)]
    pub min_reach_length_km: Option<f64>,
    /// vector-basins: keep a basin when at least this fraction of its area is inside the box.
    #[serde(default)]
    pub basin_min_inside_fraction: Option<f64>,
    /// Lakes smaller than this (km2) are not reservoirs in the fabric.
    pub lake_min_area_km2: f64,
    /// Search radius around a lake pour point, in grid cells (dem-d8) or km (vector-basins).
    pub lake_snap_radius: f64,
    /// Accept a snap only when model upstream area / lake catchment area is within [1/r, r].
    pub lake_snap_area_ratio: f64,
    /// Recipe table (channel geometry, level pool) relative to the crate data directory.
    pub recipes: String,
    /// fabric_metadata soil_class_table / veg_class_table for model_attributes.
    pub soil_class_table: String,
    pub veg_class_table: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub dataset: String,
    pub version: String,
    /// Exact object URL or tile-service pattern; null until pinned.
    pub url: Option<String>,
    pub format: String,
    /// Local path under the data root; tile inputs use the placeholders
    /// {NS} {lat2} {EW} {lon3} (south-west corner of each 1-degree tile).
    #[serde(default)]
    pub local: Option<String>,
    /// Values substituted for {region} in `local` (per-continent datasets);
    /// files that do not cover the box are skipped by their header extent.
    #[serde(default)]
    pub regions: Vec<String>,
    /// Source column -> contract meaning (e.g. "HYRIV_ID": "reach id").
    #[serde(default)]
    pub columns: BTreeMap<String, String>,
    pub licence: Licence,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaugeFeed {
    /// Agency code used in hl_uri "Gages-<agency>:<station>" (USGS keeps the
    /// bare v2.0.1 form "Gages-<station>").
    pub agency: String,
    pub service: String,
    pub url: Option<String>,
    pub licence: Licence,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Licence {
    /// "unverified" | "verified" | "conditional" | "research-only" | "excluded".
    /// Conditional and research-only inputs require a named licence opt-in.
    /// Research-only inputs never qualify for a default provider or shipment.
    pub status: String,
    pub name: Option<String>,
    /// Verbatim licence text read at the source.
    pub quote: Option<String>,
    pub url: Option<String>,
    /// UTC date the text was read.
    pub read_utc: Option<String>,
    /// Who read it (lane name).
    pub read_by: Option<String>,
}

pub fn load(path: &str) -> Result<SourceTable> {
    let t: SourceTable = serde_json::from_slice(&std::fs::read(path)?)?;
    validate(&t)?;
    Ok(t)
}

pub fn validate(t: &SourceTable) -> Result<()> {
    if t.schema != SCHEMA {
        return Err(format!("source table schema {} is not {SCHEMA}", t.schema).into());
    }
    for a in &t.auto {
        if !t.rows.iter().any(|r| r.id == a.row) {
            return Err(format!("auto rule names unknown source row {}", a.row).into());
        }
    }
    let mut ids = HashSet::new();
    let mut spaces = HashSet::new();
    for r in &t.rows {
        if !ids.insert(&r.id) {
            return Err(format!("duplicate source row {}", r.id).into());
        }
        if !(0..=900_000).contains(&r.id_namespace) || !spaces.insert(r.id_namespace) {
            return Err(format!("{}: id_namespace {} reused or out of range", r.id, r.id_namespace).into());
        }
        for need in ["river_network", "catchments", "dem", "soil", "land_cover", "lakes"] {
            if !r.inputs.contains_key(need) {
                return Err(format!("{}: inputs.{need} missing", r.id).into());
            }
        }
        for (k, i) in &r.inputs {
            check_licence(&format!("{}.inputs.{k}", r.id), &i.licence)?;
        }
        for g in &r.gauges {
            check_licence(&format!("{}.gauges.{}", r.id, g.agency), &g.licence)?;
        }
    }
    Ok(())
}

fn check_licence(at: &str, l: &Licence) -> Result<()> {
    match l.status.as_str() {
        "unverified" | "excluded" => Ok(()),
        "verified" | "conditional" | "research-only" if l.quote.as_deref().is_some_and(|q| !q.trim().is_empty()) && l.url.is_some() && l.read_utc.is_some() && l.read_by.is_some() && l.name.is_some() => Ok(()),
        "verified" | "conditional" | "research-only" => Err(format!("{at}: {} licence needs name, quote, url, read_utc and read_by", l.status).into()),
        s => Err(format!("{at}: licence status {s} is not unverified|verified|conditional|research-only|excluded").into()),
    }
}

/// Licence law: a row builds only when every input it uses was read at the
/// source and permits commercial use and redistribution of derived products.
/// Building from an unverified input would ship derived data whose terms are
/// unknown, which the licence law forbids.
pub fn admit(r: &SourceRow) -> Result<()> {
    admit_with(r, &[])
}

/// Explicit standalone cross-check admission. Naming a research-only licence
/// allows a private research build, never a default fetch or a shippable fabric
/// (coordinator ruling 2026-10-18T11:50Z). Default providers call `admit` without
/// any acceptance and therefore cannot select research-only inputs.
pub fn admit_with(r: &SourceRow, accept: &[String]) -> Result<()> {
    let ok = |l: &Licence| l.status == "verified" || (matches!(l.status.as_str(), "conditional" | "research-only") && l.name.as_ref().is_some_and(|n| accept.iter().any(|a| a == n)));
    let blocked = r
        .inputs
        .iter()
        .filter(|(_, i)| !ok(&i.licence))
        .map(|(k, i)| format!("{k}={}{}", i.licence.status, i.licence.name.as_ref().map(|n| match i.licence.status.as_str() {
            "research-only" => format!(" ({n}; limited source licence; never bundled, fetched by default or shipped; private cross-check only with --accept-licence '{n}')"),
            "conditional" => format!(" (opt in with --accept-licence '{n}')"),
            _ => format!(" ({n})"),
        }).unwrap_or_default()))
        .collect::<Vec<_>>();
    if blocked.is_empty() {
        Ok(())
    } else {
        Err(format!("{}: inputs not licence-verified ({}); derived fabric could not be redistributed", r.id, blocked.join(", ")).into())
    }
}

/// Resolve a tile input's local path for the 1-degree tile whose south-west
/// corner is (lat, lon).
pub fn tile_path(pattern: &str, lat: i32, lon: i32) -> String {
    pattern
        .replace("{NS}", if lat < 0 { "S" } else { "N" })
        .replace("{lat2}", &format!("{:02}", lat.unsigned_abs()))
        .replace("{EW}", if lon < 0 { "W" } else { "E" })
        .replace("{lon3}", &format!("{:03}", lon.unsigned_abs()))
}
