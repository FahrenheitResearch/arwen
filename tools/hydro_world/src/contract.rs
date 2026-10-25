//! Contract tables: column order, declared SQLite type, storage class, unit.
//! The declared types are exactly the ngen v2.0.1 example GeoPackage shipped at
//! ngen be11a180 (data/gauge_01073000), which is the shape the engine's reader,
//! native ngen and native t-route all parse. Two tables are HYDRO-WORLD
//! additions: `model_attributes` (engine role `model_attributes`, preserved in
//! attributes.json and derived into the CFE/Noah-OWP packets) and
//! `fabric_metadata` (provenance; the engine does not read it).
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CONTRACT: &str = "arwen.hydro-world.fabric.v1";
/// GeoPackage `PRAGMA application_id` ('GPKG') and `user_version` (1.2.0),
/// equal to the ngen example container.
pub const GPKG_APPLICATION_ID: i64 = 1_196_444_487;
pub const GPKG_USER_VERSION: i64 = 10_200;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Storage {
    Integer,
    Real,
    Text,
    Boolean,
    Geometry,
}

#[derive(Debug, Serialize)]
pub struct ColumnSpec {
    pub name: &'static str,
    /// Declared type in CREATE TABLE, verbatim.
    pub decl: &'static str,
    pub storage: Storage,
    /// Physical unit; "-" dimensionless, "id" identifier, "enum" coded value.
    pub unit: &'static str,
    /// NULL permitted in a contract-conforming fabric.
    pub nullable: bool,
    pub meaning: &'static str,
}

#[derive(Debug, Serialize)]
pub struct TableSpec {
    pub name: &'static str,
    /// gpkg_contents.data_type
    pub data_type: &'static str,
    /// gpkg_geometry_columns.geometry_type_name, for feature tables.
    pub geometry_type: Option<&'static str>,
    /// Role the engine's rw_hydrofabric reader binds this table to, if any.
    pub engine_role: Option<&'static str>,
    pub required: bool,
    pub columns: &'static [ColumnSpec],
}

macro_rules! col {
    ($n:literal, $d:literal, $s:ident, $u:literal, $null:literal, $m:literal) => {
        ColumnSpec {
            name: $n,
            decl: $d,
            storage: Storage::$s,
            unit: $u,
            nullable: $null,
            meaning: $m,
        }
    };
}

const FID: ColumnSpec = col!("fid", "INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL", Integer, "id", false, "1-based row key; equals canonical row rank");

pub const DIVIDES: &[ColumnSpec] = &[
    FID,
    col!("geom", "POLYGON", Geometry, "m (fabric CRS)", false, "catchment polygon, GPKG binary, fabric CRS"),
    col!("divide_id", "TEXT", Text, "id", false, "cat-<n>"),
    col!("toid", "TEXT", Text, "id", false, "nexus receiving this catchment's water: nex-<m> or tnx-<k>"),
    col!("type", "TEXT", Text, "enum", false, "network | terminal (v2.0.1 values)"),
    col!("ds_id", "REAL", Real, "id", true, "reserved by v2.0.1; NULL"),
    col!("areasqkm", "REAL", Real, "km2", false, "planar area in the equal-area fabric CRS"),
    col!("id", "TEXT", Text, "id", true, "wb-<n> of the paired flowpath; NULL if has_flowline=0"),
    col!("lengthkm", "REAL", Real, "km", true, "length of the paired flowpath"),
    col!("tot_drainage_areasqkm", "REAL", Real, "km2", false, "total upstream area including this catchment"),
    col!("has_flowline", "BOOLEAN", Boolean, "-", false, "1 when a flowpath drains this catchment"),
];

pub const FLOWPATHS: &[ColumnSpec] = &[
    FID,
    col!("geom", "LINESTRING", Geometry, "m (fabric CRS)", false, "reach line, upstream to downstream vertex order"),
    col!("id", "TEXT", Text, "id", false, "wb-<n>"),
    col!("toid", "TEXT", Text, "id", false, "downstream nexus nex-<m> or terminal tnx-<k>"),
    col!("mainstem", "MEDIUMINT", Integer, "id", true, "level-path identifier shared along a main stem"),
    col!("order", "REAL", Real, "-", false, "Strahler stream order"),
    col!("hydroseq", "MEDIUMINT", Integer, "-", false, "hydrologic sequence; builders write a topological rank strictly larger downstream"),
    col!("lengthkm", "REAL", Real, "km", false, "reach length (geodesic)"),
    col!("areasqkm", "REAL", Real, "km2", false, "local catchment area"),
    col!("tot_drainage_areasqkm", "REAL", Real, "km2", false, "total upstream area"),
    col!("has_divide", "BOOLEAN", Boolean, "-", false, "1 when cat-<n> exists"),
    col!("divide_id", "TEXT", Text, "id", true, "cat-<n> or NULL"),
];

pub const NEXUS: &[ColumnSpec] = &[
    FID,
    col!("geom", "POINT", Geometry, "m (fabric CRS)", false, "confluence / outlet point"),
    col!("id", "TEXT", Text, "id", false, "nex-<m> (m = n of the receiving flowpath) or tnx-<k>"),
    col!("toid", "TEXT", Text, "id", false, "wb-<m>; terminal tnx- uses the v2.0.1 sentinel wb-0"),
    col!("hl_id", "TEXT", Text, "id", true, "hydrolocation id at this nexus"),
    col!("hl_uri", "TEXT", Text, "id", false, "<hl_reference>-<hl_link>, or NA"),
    col!("type", "TEXT", Text, "enum", false, "nexus | poi (poi = carries a hydrolocation; v2.0.1 values)"),
];

pub const NETWORK: &[ColumnSpec] = &[
    FID,
    col!("id", "TEXT", Text, "id", false, "wb-<n>, nex-<m> or tnx-<k>; a flowpath repeats once per source member reach"),
    col!("toid", "TEXT", Text, "id", true, "downstream feature id"),
    col!("divide_id", "TEXT", Text, "id", true, "cat-<n> draining to this feature"),
    col!("ds_id", "REAL", Real, "id", true, "reserved by v2.0.1; NULL"),
    col!("mainstem", "MEDIUMINT", Integer, "id", true, "level-path identifier"),
    col!("hl_id", "TEXT", Text, "id", true, "hydrolocation id"),
    col!("hydroseq", "MEDIUMINT", Integer, "-", true, "hydrologic sequence"),
    col!("hl_uri", "TEXT", Text, "id", true, "Gages-<station> links a gauge (engine gauges.json)"),
    col!("hf_source", "TEXT", Text, "enum", true, "source network name (source table hf_source)"),
    col!("hf_id", "REAL", Real, "id", true, "native id of the source reach"),
    col!("lengthkm", "REAL", Real, "km", true, "reach length"),
    col!("areasqkm", "REAL", Real, "km2", true, "local catchment area"),
    col!("tot_drainage_areasqkm", "REAL", Real, "km2", true, "total upstream area"),
    col!("type", "TEXT", Text, "enum", true, "nexus | terminal | coastal | network"),
    col!("vpu", "TEXT", Text, "id", true, "processing region code"),
];

pub const FLOWPATH_ATTRIBUTES: &[ColumnSpec] = &[
    FID,
    col!("id", "TEXT", Text, "id", false, "wb-<n>; exactly one row per flowpath"),
    col!("rl_gages", "TEXT", Text, "id", true, "comma list of gauge station ids on the reach"),
    col!("rl_NHDWaterbodyComID", "TEXT", Text, "id", true, "lake hl_link the reach belongs to"),
    col!("Qi", "REAL", Real, "m3 s-1", false, "initial discharge"),
    col!("MusK", "REAL", Real, "s", false, "Muskingum K"),
    col!("MusX", "REAL", Real, "-", false, "Muskingum X"),
    col!("n", "REAL", Real, "s m-1/3", false, "Manning n, main channel (MC n)"),
    col!("So", "REAL", Real, "m m-1", false, "bed slope (MC s0), > 0"),
    col!("ChSlp", "REAL", Real, "m m-1", false, "channel side slope, horizontal per vertical (MC cs)"),
    col!("BtmWdth", "REAL", Real, "m", false, "bottom width (MC bw)"),
    col!("time", "REAL", Real, "-", true, "reserved by v2.0.1; 0"),
    col!("Kchan", "REAL", Real, "mm h-1", true, "channel loss conductance; 0 = none"),
    col!("nCC", "REAL", Real, "s m-1/3", false, "Manning n, compound channel (MC ncc)"),
    col!("TopWdthCC", "REAL", Real, "m", false, "compound channel top width (MC twcc)"),
    col!("TopWdth", "REAL", Real, "m", false, "bankfull top width (MC tw)"),
    col!("length_m", "REAL", Real, "m", false, "reach length (MC dx); = 1000 * flowpaths.lengthkm"),
];

pub const HYDROLOCATIONS: &[ColumnSpec] = &[
    FID,
    col!("geom", "POINT", Geometry, "m (fabric CRS)", false, "location point"),
    col!("hl_id", "TEXT", Text, "id", false, "hydrolocation id, unique in fabric"),
    col!("hl_reference", "TEXT", Text, "enum", false, "Gages | WBOut | WBIn | HUC12 | ..."),
    col!("hl_link", "TEXT", Text, "id", false, "native id in hl_reference's namespace"),
    col!("hl_uri", "TEXT", Text, "id", false, "<hl_reference>-<hl_link>"),
    col!("hl_position", "TEXT", Text, "enum", false, "outflow | inflow"),
    col!("id", "TEXT", Text, "id", false, "nexus id the location sits on"),
];

pub const LAKES: &[ColumnSpec] = &[
    FID,
    col!("geom", "GEOMETRY", Geometry, "m (fabric CRS)", true, "outlet point"),
    col!("id", "TEXT", Text, "id", false, "outlet nexus nex-<m>"),
    col!("toid", "TEXT", Text, "id", false, "outlet flowpath wb-<m>"),
    col!("hl_id", "TEXT", Text, "id", true, "hydrolocation id"),
    col!("hl_reference", "TEXT", Text, "enum", false, "WBOut"),
    col!("hl_link", "REAL", Real, "id", false, "waterbody id (t-route lake id); integer valued"),
    col!("hl_uri", "TEXT", Text, "id", false, "WBOut-<hl_link>"),
    col!("Dam_Length", "REAL", Real, "m", false, "dam length (t-route dam_length_m)"),
    col!("ifd", "REAL", Real, "-", false, "initial fractional depth (0..1)"),
    col!("LkArea", "REAL", Real, "km2", false, "lake area (area_km2)"),
    col!("LkMxE", "REAL", Real, "m (fabric vertical datum)", false, "maximum water elevation (maximum_m)"),
    col!("OrificeA", "REAL", Real, "m2", false, "orifice area"),
    col!("OrificeC", "REAL", Real, "-", false, "orifice coefficient"),
    col!("OrificeE", "REAL", Real, "m (fabric vertical datum)", false, "orifice elevation"),
    col!("time", "REAL", Real, "-", true, "reserved by v2.0.1; 0"),
    col!("WeirC", "REAL", Real, "-", false, "weir coefficient"),
    col!("WeirE", "REAL", Real, "m (fabric vertical datum)", false, "weir crest elevation"),
    col!("WeirL", "REAL", Real, "m", false, "weir length"),
];

pub const MODEL_ATTRIBUTES: &[ColumnSpec] = &[
    FID,
    col!("divide_id", "TEXT", Text, "id", false, "cat-<n>; one row per divide, divides order"),
    col!("elevation_mean_m", "REAL", Real, "m (fabric vertical datum)", false, "mean DEM elevation"),
    col!("slope_mean_deg", "REAL", Real, "deg", false, "mean terrain slope (Noah-OWP terrain_slope)"),
    col!("aspect_mean_deg", "REAL", Real, "deg clockwise from north", false, "circular mean aspect (Noah-OWP azimuth)"),
    col!("centroid_lat_deg", "REAL", Real, "deg north, WGS84", false, "Noah-OWP lat"),
    col!("centroid_lon_deg", "REAL", Real, "deg east, WGS84", false, "Noah-OWP lon"),
    col!("soil_class", "INTEGER", Integer, "enum", false, "index into fabric_metadata soil_class_table (Noah-OWP isltyp)"),
    col!("veg_class", "INTEGER", Integer, "enum", false, "index into fabric_metadata veg_class_table (Noah-OWP vegtyp)"),
    col!("soil_color", "INTEGER", Integer, "enum", false, "Noah-OWP soilcolor (1..20)"),
    col!("impervious_fraction", "REAL", Real, "-", false, "0..1 (CFE urban_decimal_fraction)"),
    col!("soil_depth_m", "REAL", Real, "m", false, "CFE soil_params.depth"),
    col!("soil_b", "REAL", Real, "-", false, "CFE soil_params.b (Clapp-Hornberger b)"),
    col!("soil_satdk_m_s", "REAL", Real, "m s-1", false, "CFE soil_params.satdk"),
    col!("soil_satpsi_m", "REAL", Real, "m", false, "CFE soil_params.satpsi"),
    col!("soil_slop", "REAL", Real, "m m-1", false, "CFE soil_params.slop"),
    col!("soil_smcmax", "REAL", Real, "m3 m-3", false, "CFE soil_params.smcmax"),
    col!("soil_wltsmc", "REAL", Real, "m3 m-3", false, "CFE soil_params.wltsmc"),
    col!("refkdt", "REAL", Real, "-", false, "CFE refkdt"),
    col!("alpha_fc", "REAL", Real, "-", false, "CFE alpha_fc"),
    col!("max_gw_storage_m", "REAL", Real, "m", false, "CFE max_gw_storage"),
    col!("cgw_m_h", "REAL", Real, "m h-1", false, "CFE Cgw"),
    col!("gw_expon", "REAL", Real, "-", false, "CFE expon"),
    col!("k_lf", "REAL", Real, "-", false, "CFE K_lf"),
    col!("k_nash", "REAL", Real, "-", false, "CFE K_nash"),
    col!("giuh_ordinates", "TEXT", Text, "- per CFE step", false, "comma list summing to 1 (CFE giuh_ordinates)"),
];

pub const FABRIC_METADATA: &[ColumnSpec] = &[
    FID,
    col!("key", "TEXT", Text, "id", false, "metadata key (see contract doc)"),
    col!("value", "TEXT", Text, "-", false, "metadata value"),
];

pub const TABLES: &[TableSpec] = &[
    TableSpec { name: "divides", data_type: "features", geometry_type: Some("POLYGON"), engine_role: Some("divides"), required: true, columns: DIVIDES },
    TableSpec { name: "flowpaths", data_type: "features", geometry_type: Some("LINESTRING"), engine_role: Some("flowpaths"), required: true, columns: FLOWPATHS },
    TableSpec { name: "nexus", data_type: "features", geometry_type: Some("POINT"), engine_role: Some("nexus"), required: true, columns: NEXUS },
    TableSpec { name: "network", data_type: "attributes", geometry_type: None, engine_role: Some("network"), required: true, columns: NETWORK },
    TableSpec { name: "flowpath_attributes", data_type: "attributes", geometry_type: None, engine_role: Some("flowpath_attributes"), required: true, columns: FLOWPATH_ATTRIBUTES },
    TableSpec { name: "hydrolocations", data_type: "features", geometry_type: Some("POINT"), engine_role: Some("hydrolocations"), required: true, columns: HYDROLOCATIONS },
    TableSpec { name: "lakes", data_type: "features", geometry_type: Some("GEOMETRY"), engine_role: Some("lakes"), required: true, columns: LAKES },
    TableSpec { name: "model_attributes", data_type: "attributes", geometry_type: None, engine_role: Some("model_attributes"), required: true, columns: MODEL_ATTRIBUTES },
    TableSpec { name: "fabric_metadata", data_type: "attributes", geometry_type: None, engine_role: None, required: true, columns: FABRIC_METADATA },
];

pub fn table(name: &str) -> Option<&'static TableSpec> {
    TABLES.iter().find(|t| t.name == name)
}

/// Required `fabric_metadata` keys for a builder-produced fabric.
pub const METADATA_KEYS: &[&str] = &[
    "contract",
    "builder",
    "builder_revision",
    "source_row",
    "source_table_sha256",
    "data_terms",
    "id_namespace",
    "horizontal_crs",
    "vertical_datum",
    "soil_class_table",
    "veg_class_table",
    "domain",
    "build_utc",
];

/// GeoPackage binary geometry, carried verbatim (header + little-endian WKB).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry(pub Vec<u8>);
impl Serialize for Geometry {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        const HEX: &[u8;16] = b"0123456789abcdef";
        let mut h = String::with_capacity(self.0.len() * 2);
        for &b in &self.0 {
            h.push(HEX[(b >> 4) as usize] as char);
            h.push(HEX[(b & 15) as usize] as char);
        }
        s.serialize_str(&h)
    }
}
impl<'de> Deserialize<'de> for Geometry {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let h = String::deserialize(d)?;
        if h.len() % 2 != 0 {
            return Err(serde::de::Error::custom("odd hex geometry length"));
        }
        let nibble = |b:u8| match b {
            b'0'..=b'9' => Some(b-b'0'), b'a'..=b'f' => Some(b-b'a'+10),
            b'A'..=b'F' => Some(b-b'A'+10), _ => None,
        };
        h.as_bytes().chunks_exact(2).map(|b|
            Some((nibble(b[0])? << 4) | nibble(b[1])?))
            .collect::<Option<Vec<_>>>().map(Geometry)
            .ok_or_else(||serde::de::Error::custom("invalid hex geometry"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Divide {
    pub fid: i64,
    pub geom: Option<Geometry>,
    pub divide_id: Option<String>,
    pub toid: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub ds_id: Option<f64>,
    pub areasqkm: Option<f64>,
    pub id: Option<String>,
    pub lengthkm: Option<f64>,
    pub tot_drainage_areasqkm: Option<f64>,
    pub has_flowline: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Flowpath {
    pub fid: i64,
    pub geom: Option<Geometry>,
    pub id: Option<String>,
    pub toid: Option<String>,
    pub mainstem: Option<i64>,
    pub order: Option<f64>,
    pub hydroseq: Option<i64>,
    pub lengthkm: Option<f64>,
    pub areasqkm: Option<f64>,
    pub tot_drainage_areasqkm: Option<f64>,
    pub has_divide: Option<bool>,
    pub divide_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nexus {
    pub fid: i64,
    pub geom: Option<Geometry>,
    pub id: Option<String>,
    pub toid: Option<String>,
    pub hl_id: Option<String>,
    pub hl_uri: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkRow {
    pub fid: i64,
    pub id: Option<String>,
    pub toid: Option<String>,
    pub divide_id: Option<String>,
    pub ds_id: Option<f64>,
    pub mainstem: Option<i64>,
    pub hl_id: Option<String>,
    pub hydroseq: Option<i64>,
    pub hl_uri: Option<String>,
    pub hf_source: Option<String>,
    pub hf_id: Option<f64>,
    pub lengthkm: Option<f64>,
    pub areasqkm: Option<f64>,
    pub tot_drainage_areasqkm: Option<f64>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub vpu: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowpathAttributes {
    pub fid: i64,
    pub id: Option<String>,
    pub rl_gages: Option<String>,
    #[serde(rename = "rl_NHDWaterbodyComID")]
    pub rl_nhd_waterbody_comid: Option<String>,
    #[serde(rename = "Qi")]
    pub qi: Option<f64>,
    #[serde(rename = "MusK")]
    pub mus_k: Option<f64>,
    #[serde(rename = "MusX")]
    pub mus_x: Option<f64>,
    pub n: Option<f64>,
    #[serde(rename = "So")]
    pub so: Option<f64>,
    #[serde(rename = "ChSlp")]
    pub ch_slp: Option<f64>,
    #[serde(rename = "BtmWdth")]
    pub btm_wdth: Option<f64>,
    pub time: Option<f64>,
    #[serde(rename = "Kchan")]
    pub kchan: Option<f64>,
    #[serde(rename = "nCC")]
    pub n_cc: Option<f64>,
    #[serde(rename = "TopWdthCC")]
    pub top_wdth_cc: Option<f64>,
    #[serde(rename = "TopWdth")]
    pub top_wdth: Option<f64>,
    pub length_m: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hydrolocation {
    pub fid: i64,
    pub geom: Option<Geometry>,
    pub hl_id: Option<String>,
    pub hl_reference: Option<String>,
    pub hl_link: Option<String>,
    pub hl_uri: Option<String>,
    pub hl_position: Option<String>,
    pub id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lake {
    pub fid: i64,
    pub geom: Option<Geometry>,
    pub id: Option<String>,
    pub toid: Option<String>,
    pub hl_id: Option<String>,
    pub hl_reference: Option<String>,
    pub hl_link: Option<f64>,
    pub hl_uri: Option<String>,
    #[serde(rename = "Dam_Length")]
    pub dam_length: Option<f64>,
    pub ifd: Option<f64>,
    #[serde(rename = "LkArea")]
    pub lk_area: Option<f64>,
    #[serde(rename = "LkMxE")]
    pub lk_mx_e: Option<f64>,
    #[serde(rename = "OrificeA")]
    pub orifice_a: Option<f64>,
    #[serde(rename = "OrificeC")]
    pub orifice_c: Option<f64>,
    #[serde(rename = "OrificeE")]
    pub orifice_e: Option<f64>,
    pub time: Option<f64>,
    #[serde(rename = "WeirC")]
    pub weir_c: Option<f64>,
    #[serde(rename = "WeirE")]
    pub weir_e: Option<f64>,
    #[serde(rename = "WeirL")]
    pub weir_l: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAttributes {
    pub fid: i64,
    pub divide_id: Option<String>,
    pub elevation_mean_m: Option<f64>,
    pub slope_mean_deg: Option<f64>,
    pub aspect_mean_deg: Option<f64>,
    pub centroid_lat_deg: Option<f64>,
    pub centroid_lon_deg: Option<f64>,
    pub soil_class: Option<i64>,
    pub veg_class: Option<i64>,
    pub soil_color: Option<i64>,
    pub impervious_fraction: Option<f64>,
    pub soil_depth_m: Option<f64>,
    pub soil_b: Option<f64>,
    pub soil_satdk_m_s: Option<f64>,
    pub soil_satpsi_m: Option<f64>,
    pub soil_slop: Option<f64>,
    pub soil_smcmax: Option<f64>,
    pub soil_wltsmc: Option<f64>,
    pub refkdt: Option<f64>,
    pub alpha_fc: Option<f64>,
    pub max_gw_storage_m: Option<f64>,
    pub cgw_m_h: Option<f64>,
    pub gw_expon: Option<f64>,
    pub k_lf: Option<f64>,
    pub k_nash: Option<f64>,
    pub giuh_ordinates: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetadataRow {
    pub fid: i64,
    pub key: Option<String>,
    pub value: Option<String>,
}

/// One `gpkg_spatial_ref_sys` row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Srs {
    pub srs_name: String,
    pub srs_id: i64,
    pub organization: String,
    pub organization_coordsys_id: i64,
    pub definition: String,
    pub description: Option<String>,
}

/// One `gpkg_contents` row (minus table_name/data_type/srs, which the spec owns).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Contents {
    pub identifier: Option<String>,
    pub description: Option<String>,
    pub last_change: String,
    pub bbox: Option<[f64; 4]>,
}

/// A whole fabric in contract form. Tables absent from a source (pre-contract
/// NOAA files carry no model_attributes/fabric_metadata) are `None`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fabric {
    pub contract: String,
    pub srs: Srs,
    pub contents: BTreeMap<String, Contents>,
    pub divides: Vec<Divide>,
    pub flowpaths: Vec<Flowpath>,
    pub nexus: Vec<Nexus>,
    pub network: Vec<NetworkRow>,
    pub flowpath_attributes: Vec<FlowpathAttributes>,
    pub hydrolocations: Vec<Hydrolocation>,
    pub lakes: Vec<Lake>,
    pub model_attributes: Option<Vec<ModelAttributes>>,
    pub fabric_metadata: Option<Vec<MetadataRow>>,
}
