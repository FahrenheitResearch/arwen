use crate::shared_context::DomainSpec;
use rustwx_core::GridProjection;
use rustwx_render::{
    Color, MapRenderRequest, ProjectedDomainBuildOptions, ProjectedLabelPlacement,
    ProjectedPlaceLabel, ProjectedPlaceLabelStyle, ProjectionSpec, build_projected_domain,
};
use serde::{Deserialize, Serialize};

const KM_PER_DEG_LAT: f64 = 111.32;
pub const DEFAULT_PLACE_HALF_HEIGHT_DEG: f64 = 1.9;
pub const AUX_PLACE_HALF_HEIGHT_DEG: f64 = 1.05;
pub const MICRO_PLACE_HALF_HEIGHT_DEG: f64 = 0.58;
pub const PLACE_OUTPUT_ASPECT_RATIO: f64 = 1200.0 / 900.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlacePreset {
    pub slug: &'static str,
    pub label: &'static str,
    pub center_lon: f64,
    pub center_lat: f64,
    pub half_height_deg: f64,
}

pub type MetroCropPreset = PlacePreset;

const REGION_PLACE_LABEL_SLUGS: &[&str] = &[
    "conus",
    "midwest",
    "california",
    "california_square",
    "reno_square",
    "pacific_northwest",
    "california_southwest",
    "rockies_high_plains",
    "southeast",
    "southern_plains",
    "oklahoma",
    "northeast",
    "great_lakes",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaceLabelDomainKind {
    Conus,
    Region,
    CityCrop,
}

#[derive(Debug, Clone, Copy)]
struct PlaceLabelPlan {
    kind: PlaceLabelDomainKind,
    max_count: usize,
    min_center_spacing_km: f64,
    max_crop_overlap_fraction: f64,
    anchor_slug: Option<&'static str>,
}

#[derive(Debug, Clone)]
struct RankedPlaceCandidate {
    place: SelectedPlace,
    selection_bounds: (f64, f64, f64, f64),
}

impl PlacePreset {
    pub fn bounds(self) -> (f64, f64, f64, f64) {
        centered_bounds(
            self.center_lon,
            self.center_lat,
            self.half_height_deg,
            PLACE_OUTPUT_ASPECT_RATIO,
        )
    }

    pub fn domain(self) -> DomainSpec {
        DomainSpec::new(self.slug, self.bounds())
    }

    pub fn center(self) -> (f64, f64) {
        (self.center_lon, self.center_lat)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaceContainmentMode {
    #[default]
    CenterWithinBounds,
    CropIntersectsBounds,
    CropWithinBounds,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlaceSelectionOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_count: Option<usize>,
    pub min_center_spacing_km: f64,
    pub max_crop_overlap_fraction: f64,
    #[serde(default)]
    pub containment: PlaceContainmentMode,
}

impl Default for PlaceSelectionOptions {
    fn default() -> Self {
        Self::for_overlay_labels()
    }
}

impl PlaceSelectionOptions {
    pub fn for_overlay_labels() -> Self {
        Self {
            max_count: None,
            min_center_spacing_km: 220.0,
            max_crop_overlap_fraction: 0.35,
            containment: PlaceContainmentMode::CropIntersectsBounds,
        }
    }

    pub fn for_city_crops() -> Self {
        Self {
            max_count: None,
            min_center_spacing_km: 0.0,
            max_crop_overlap_fraction: 1.0,
            containment: PlaceContainmentMode::CenterWithinBounds,
        }
    }

    pub fn with_max_count(mut self, max_count: usize) -> Self {
        self.max_count = Some(max_count);
        self
    }

    pub fn with_min_center_spacing_km(mut self, min_center_spacing_km: f64) -> Self {
        self.min_center_spacing_km = min_center_spacing_km.max(0.0);
        self
    }

    pub fn with_max_crop_overlap_fraction(mut self, max_crop_overlap_fraction: f64) -> Self {
        self.max_crop_overlap_fraction = max_crop_overlap_fraction.clamp(0.0, 1.0);
        self
    }

    pub fn with_containment(mut self, containment: PlaceContainmentMode) -> Self {
        self.containment = containment;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedPlace {
    pub slug: String,
    pub label: String,
    pub center_lon: f64,
    pub center_lat: f64,
    pub bounds: (f64, f64, f64, f64),
    pub source_index: usize,
    pub center_distance_km: f64,
    pub edge_margin_km: f64,
    pub ranking_score: f64,
}

impl SelectedPlace {
    pub fn domain(&self) -> DomainSpec {
        DomainSpec::new(self.slug.clone(), self.bounds)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PlaceLabelOverlay {
    #[serde(
        default,
        skip_serializing_if = "PlaceLabelDensityTier::is_default_major"
    )]
    pub density: PlaceLabelDensityTier,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub included_place_slugs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlaceLabelDensityTier {
    None,
    #[default]
    Major,
    MajorAndAux,
    Dense,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PlaceCatalogTier {
    Major,
    Aux,
    Micro,
}

impl PlaceLabelDensityTier {
    pub fn from_numeric(value: u8) -> Self {
        match value {
            0 => Self::None,
            1 => Self::Major,
            2 => Self::MajorAndAux,
            _ => Self::Dense,
        }
    }

    pub fn as_numeric(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Major => 1,
            Self::MajorAndAux => 2,
            Self::Dense => 3,
        }
    }

    fn is_default_major(&self) -> bool {
        matches!(self, Self::Major)
    }
}

impl PlaceLabelOverlay {
    pub fn none() -> Self {
        Self {
            density: PlaceLabelDensityTier::None,
            included_place_slugs: Vec::new(),
        }
    }

    pub fn major_us_cities() -> Self {
        Self {
            density: PlaceLabelDensityTier::Major,
            included_place_slugs: Vec::new(),
        }
    }

    pub fn with_density(mut self, density: PlaceLabelDensityTier) -> Self {
        self.density = density;
        self
    }

    pub fn with_included_place_slugs<I, S>(mut self, slugs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.included_place_slugs.clear();
        for slug in slugs {
            let slug = slug.into();
            if slug.is_empty()
                || self
                    .included_place_slugs
                    .iter()
                    .any(|existing| existing == &slug)
            {
                continue;
            }
            self.included_place_slugs.push(slug);
        }
        self
    }

    pub fn includes_place(&self, slug: &str) -> bool {
        self.included_place_slugs.is_empty()
            || self
                .included_place_slugs
                .iter()
                .any(|included| included == slug)
    }

    pub fn selected_places_for_domain(&self, domain: &DomainSpec) -> Vec<SelectedPlace> {
        let Some(plan) = place_label_plan_for_domain(domain) else {
            return Vec::new();
        };
        select_places_for_label_plan(domain, plan, self)
    }

    fn filtered_catalog(&self) -> Vec<PlacePreset> {
        match self.density {
            PlaceLabelDensityTier::None => Vec::new(),
            PlaceLabelDensityTier::Major => MAJOR_US_CITY_PRESETS
                .iter()
                .copied()
                .filter(|preset| self.includes_place(preset.slug))
                .collect(),
            PlaceLabelDensityTier::MajorAndAux => MAJOR_US_CITY_PRESETS
                .iter()
                .chain(AUX_US_CITY_PRESETS.iter())
                .copied()
                .filter(|preset| self.includes_place(preset.slug))
                .collect(),
            PlaceLabelDensityTier::Dense => MAJOR_US_CITY_PRESETS
                .iter()
                .chain(AUX_US_CITY_PRESETS.iter())
                .chain(MICRO_US_PLACE_PRESETS.iter())
                .copied()
                .filter(|preset| self.includes_place(preset.slug))
                .collect(),
        }
    }
}

pub const MAJOR_US_CITY_PRESETS: &[PlacePreset] = &[
    place("al_birmingham", "Birmingham, AL", -86.80, 33.52),
    place("az_phoenix", "Phoenix, AZ", -112.07, 33.45),
    place("ar_little_rock", "Little Rock, AR", -92.29, 34.75),
    place("ca_los_angeles", "Los Angeles, CA", -118.24, 34.05),
    place(
        "ca_san_francisco_bay",
        "San Francisco Bay, CA",
        -122.27,
        37.80,
    ),
    place("ca_sacramento", "Sacramento, CA", -121.49, 38.58),
    place("ca_san_diego", "San Diego, CA", -117.16, 32.72),
    place("co_denver", "Denver, CO", -104.99, 39.74),
    place("ct_hartford", "Hartford, CT", -72.67, 41.77),
    place("de_wilmington", "Wilmington, DE", -75.55, 39.74),
    place("dc_washington", "Washington, DC", -77.04, 38.91),
    place("fl_miami", "Miami, FL", -80.19, 25.76),
    place("fl_tampa", "Tampa, FL", -82.46, 27.95),
    place("fl_orlando", "Orlando, FL", -81.38, 28.54),
    place("ga_atlanta", "Atlanta, GA", -84.39, 33.75),
    place("id_boise", "Boise, ID", -116.20, 43.62),
    place("il_chicago", "Chicago, IL", -87.63, 41.88),
    place("in_indianapolis", "Indianapolis, IN", -86.16, 39.77),
    place("ia_des_moines", "Des Moines, IA", -93.62, 41.59),
    place("ks_wichita", "Wichita, KS", -97.34, 37.69),
    place("ky_louisville", "Louisville, KY", -85.76, 38.25),
    place("la_new_orleans", "New Orleans, LA", -90.07, 29.95),
    place("me_portland", "Portland, ME", -70.26, 43.66),
    place("md_baltimore", "Baltimore, MD", -76.61, 39.29),
    place("ma_boston", "Boston, MA", -71.06, 42.36),
    place("mi_detroit", "Detroit, MI", -83.05, 42.33),
    place("mn_minneapolis", "Minneapolis, MN", -93.27, 44.98),
    place("ms_jackson", "Jackson, MS", -90.18, 32.30),
    place("mo_st_louis", "St. Louis, MO", -90.20, 38.63),
    place("mt_billings", "Billings, MT", -108.50, 45.78),
    place("ne_omaha", "Omaha, NE", -95.94, 41.26),
    place("nv_las_vegas", "Las Vegas, NV", -115.14, 36.17),
    place("nv_reno", "Reno, NV", -119.81, 39.53),
    place("nh_manchester", "Manchester, NH", -71.45, 42.99),
    place("nj_newark", "Newark, NJ", -74.17, 40.74),
    place("nm_albuquerque", "Albuquerque, NM", -106.65, 35.08),
    place("ny_new_york_city", "New York City, NY", -74.00, 40.71),
    place("nc_charlotte", "Charlotte, NC", -80.84, 35.23),
    place("nd_fargo", "Fargo, ND", -96.79, 46.88),
    place("oh_columbus", "Columbus, OH", -82.99, 39.96),
    place("ok_oklahoma_city", "Oklahoma City, OK", -97.52, 35.47),
    place("or_portland", "Portland, OR", -122.68, 45.52),
    place("pa_philadelphia", "Philadelphia, PA", -75.17, 39.95),
    place("ri_providence", "Providence, RI", -71.41, 41.82),
    place("sc_charleston", "Charleston, SC", -79.93, 32.78),
    place("sd_sioux_falls", "Sioux Falls, SD", -96.73, 43.55),
    place("tn_nashville", "Nashville, TN", -86.78, 36.16),
    place(
        "tx_dallas_fort_worth",
        "Dallas-Fort Worth, TX",
        -97.04,
        32.90,
    ),
    place("tx_houston", "Houston, TX", -95.37, 29.76),
    place("tx_austin", "Austin, TX", -97.74, 30.27),
    place("tx_san_antonio", "San Antonio, TX", -98.49, 29.42),
    place("ut_salt_lake_city", "Salt Lake City, UT", -111.89, 40.76),
    place("vt_burlington", "Burlington, VT", -73.21, 44.48),
    place("va_richmond", "Richmond, VA", -77.44, 37.54),
    place("wa_seattle", "Seattle, WA", -122.33, 47.61),
    place("wv_charleston", "Charleston, WV", -81.63, 38.35),
    place("wi_milwaukee", "Milwaukee, WI", -87.91, 43.04),
    place("wy_cheyenne", "Cheyenne, WY", -104.82, 41.14),
];

pub const AUX_US_CITY_PRESETS: &[PlacePreset] = &[
    place("al_huntsville", "Huntsville, AL", -86.59, 34.73),
    place("al_mobile", "Mobile, AL", -88.04, 30.69),
    place("al_montgomery", "Montgomery, AL", -86.30, 32.37),
    place("ar_fayetteville", "Fayetteville, AR", -94.16, 36.06),
    place("az_tucson", "Tucson, AZ", -110.97, 32.22),
    place("ca_bakersfield", "Bakersfield, CA", -119.02, 35.37),
    place("ca_fresno", "Fresno, CA", -119.79, 36.74),
    place("ca_monterey", "Monterey, CA", -121.89, 36.60),
    place("ca_palm_springs", "Palm Springs, CA", -116.55, 33.83),
    place("ca_redding", "Redding, CA", -122.39, 40.58),
    place("ca_san_jose", "San Jose, CA", -121.89, 37.34),
    place("ca_san_luis_obispo", "San Luis Obispo, CA", -120.66, 35.28),
    place("ca_santa_barbara", "Santa Barbara, CA", -119.70, 34.42),
    place("ca_santa_rosa", "Santa Rosa, CA", -122.71, 38.44),
    place(
        "co_colorado_springs",
        "Colorado Springs, CO",
        -104.82,
        38.83,
    ),
    place("co_fort_collins", "Fort Collins, CO", -105.08, 40.59),
    place("co_grand_junction", "Grand Junction, CO", -108.55, 39.06),
    place("co_pueblo", "Pueblo, CO", -104.61, 38.25),
    place("ct_new_haven", "New Haven, CT", -72.93, 41.31),
    place("de_dover", "Dover, DE", -75.52, 39.16),
    place("fl_fort_myers", "Fort Myers, FL", -81.87, 26.64),
    place("fl_jacksonville", "Jacksonville, FL", -81.66, 30.33),
    place("fl_pensacola", "Pensacola, FL", -87.22, 30.42),
    place("fl_tallahassee", "Tallahassee, FL", -84.28, 30.44),
    place("fl_west_palm_beach", "West Palm Beach, FL", -80.05, 26.71),
    place("ga_augusta", "Augusta, GA", -81.97, 33.47),
    place("ga_macon", "Macon, GA", -83.63, 32.84),
    place("ga_savannah", "Savannah, GA", -81.10, 32.08),
    place("ia_cedar_rapids", "Cedar Rapids, IA", -91.67, 41.98),
    place("ia_davenport", "Davenport, IA", -90.58, 41.52),
    place("ia_sioux_city", "Sioux City, IA", -96.40, 42.50),
    place("id_coeur_dalene", "Coeur d'Alene, ID", -116.78, 47.68),
    place("id_idaho_falls", "Idaho Falls, ID", -112.03, 43.49),
    place("id_pocatello", "Pocatello, ID", -112.45, 42.87),
    place("il_peoria", "Peoria, IL", -89.59, 40.69),
    place("il_rockford", "Rockford, IL", -89.09, 42.27),
    place("il_springfield", "Springfield, IL", -89.65, 39.78),
    place("in_evansville", "Evansville, IN", -87.57, 37.97),
    place("in_fort_wayne", "Fort Wayne, IN", -85.14, 41.08),
    place("in_south_bend", "South Bend, IN", -86.25, 41.68),
    place("ks_dodge_city", "Dodge City, KS", -100.02, 37.75),
    place("ks_salina", "Salina, KS", -97.61, 38.84),
    place("ks_topeka", "Topeka, KS", -95.68, 39.05),
    place("ky_bowling_green", "Bowling Green, KY", -86.44, 36.97),
    place("ky_lexington", "Lexington, KY", -84.50, 38.05),
    place("ky_paducah", "Paducah, KY", -88.60, 37.08),
    place("la_baton_rouge", "Baton Rouge, LA", -91.14, 30.45),
    place("la_lafayette", "Lafayette, LA", -92.02, 30.22),
    place("la_lake_charles", "Lake Charles, LA", -93.22, 30.23),
    place("la_shreveport", "Shreveport, LA", -93.75, 32.53),
    place("ma_springfield", "Springfield, MA", -72.59, 42.10),
    place("ma_worcester", "Worcester, MA", -71.80, 42.26),
    place("md_frederick", "Frederick, MD", -77.41, 39.41),
    place("me_bangor", "Bangor, ME", -68.78, 44.80),
    place("mi_grand_rapids", "Grand Rapids, MI", -85.67, 42.96),
    place("mi_lansing", "Lansing, MI", -84.56, 42.73),
    place("mi_traverse_city", "Traverse City, MI", -85.62, 44.76),
    place("mn_duluth", "Duluth, MN", -92.10, 46.79),
    place("mn_rochester", "Rochester, MN", -92.47, 44.01),
    place("mn_st_cloud", "St. Cloud, MN", -94.16, 45.56),
    place("mo_columbia", "Columbia, MO", -92.33, 38.95),
    place("mo_kansas_city", "Kansas City, MO", -94.58, 39.10),
    place("mo_springfield", "Springfield, MO", -93.29, 37.21),
    place("ms_gulfport", "Gulfport, MS", -89.09, 30.37),
    place("ms_hattiesburg", "Hattiesburg, MS", -89.29, 31.33),
    place("ms_tupelo", "Tupelo, MS", -88.70, 34.26),
    place("mt_bozeman", "Bozeman, MT", -111.04, 45.68),
    place("mt_great_falls", "Great Falls, MT", -111.30, 47.50),
    place("mt_missoula", "Missoula, MT", -113.99, 46.87),
    place("nc_asheville", "Asheville, NC", -82.55, 35.60),
    place("nc_greensboro", "Greensboro, NC", -79.79, 36.07),
    place("nc_raleigh", "Raleigh, NC", -78.64, 35.78),
    place("nc_wilmington", "Wilmington, NC", -77.94, 34.23),
    place("nd_bismarck", "Bismarck, ND", -100.78, 46.81),
    place("nd_grand_forks", "Grand Forks, ND", -97.03, 47.93),
    place("ne_lincoln", "Lincoln, NE", -96.68, 40.81),
    place("ne_north_platte", "North Platte, NE", -100.77, 41.14),
    place("ne_scottsbluff", "Scottsbluff, NE", -103.67, 41.87),
    place("nh_portsmouth", "Portsmouth, NH", -70.76, 43.07),
    place("nj_atlantic_city", "Atlantic City, NJ", -74.42, 39.36),
    place("nm_roswell", "Roswell, NM", -104.52, 33.39),
    place("nm_santa_fe", "Santa Fe, NM", -105.94, 35.69),
    place("nv_elko", "Elko, NV", -115.76, 40.83),
    place("ny_albany", "Albany, NY", -73.75, 42.65),
    place("ny_binghamton", "Binghamton, NY", -75.91, 42.10),
    place("ny_buffalo", "Buffalo, NY", -78.88, 42.89),
    place("ny_rochester", "Rochester, NY", -77.61, 43.16),
    place("ny_syracuse", "Syracuse, NY", -76.15, 43.05),
    place("oh_cincinnati", "Cincinnati, OH", -84.51, 39.10),
    place("oh_cleveland", "Cleveland, OH", -81.69, 41.50),
    place("oh_dayton", "Dayton, OH", -84.19, 39.76),
    place("oh_toledo", "Toledo, OH", -83.55, 41.65),
    place("ok_enid", "Enid, OK", -97.88, 36.40),
    place("ok_lawton", "Lawton, OK", -98.40, 34.61),
    place("ok_tulsa", "Tulsa, OK", -95.99, 36.15),
    place("or_bend", "Bend, OR", -121.31, 44.06),
    place("or_eugene", "Eugene, OR", -123.09, 44.05),
    place("or_medford", "Medford, OR", -122.87, 42.33),
    place("pa_erie", "Erie, PA", -80.09, 42.13),
    place("pa_harrisburg", "Harrisburg, PA", -76.88, 40.27),
    place("pa_pittsburgh", "Pittsburgh, PA", -79.99, 40.44),
    place("pa_state_college", "State College, PA", -77.86, 40.79),
    place("ri_newport", "Newport, RI", -71.31, 41.49),
    place("sc_columbia", "Columbia, SC", -81.03, 34.00),
    place("sc_greenville", "Greenville, SC", -82.40, 34.85),
    place("sc_myrtle_beach", "Myrtle Beach, SC", -78.88, 33.69),
    place("sd_pierre", "Pierre, SD", -100.35, 44.37),
    place("sd_rapid_city", "Rapid City, SD", -103.23, 44.08),
    place("tn_chattanooga", "Chattanooga, TN", -85.31, 35.05),
    place("tn_knoxville", "Knoxville, TN", -83.92, 35.96),
    place("tn_memphis", "Memphis, TN", -90.05, 35.15),
    place("tx_abilene", "Abilene, TX", -99.74, 32.45),
    place("tx_amarillo", "Amarillo, TX", -101.83, 35.22),
    place("tx_beaumont", "Beaumont, TX", -94.13, 30.08),
    place("tx_brownsville", "Brownsville, TX", -97.50, 25.90),
    place("tx_corpus_christi", "Corpus Christi, TX", -97.40, 27.80),
    place("tx_el_paso", "El Paso, TX", -106.49, 31.76),
    place("tx_lubbock", "Lubbock, TX", -101.86, 33.58),
    place("tx_midland", "Midland, TX", -102.08, 31.99),
    place("tx_waco", "Waco, TX", -97.15, 31.55),
    place("ut_moab", "Moab, UT", -109.55, 38.57),
    place("ut_ogden", "Ogden, UT", -111.97, 41.22),
    place("ut_st_george", "St. George, UT", -113.58, 37.10),
    place("va_charlottesville", "Charlottesville, VA", -78.48, 38.03),
    place("va_norfolk", "Norfolk, VA", -76.29, 36.85),
    place("va_roanoke", "Roanoke, VA", -79.94, 37.27),
    place("vt_montpelier", "Montpelier, VT", -72.58, 44.26),
    place("wa_bellingham", "Bellingham, WA", -122.48, 48.75),
    place("wa_olympia", "Olympia, WA", -122.90, 47.04),
    place("wa_spokane", "Spokane, WA", -117.43, 47.66),
    place("wa_yakima", "Yakima, WA", -120.51, 46.60),
    place("wi_eau_claire", "Eau Claire, WI", -91.50, 44.81),
    place("wi_green_bay", "Green Bay, WI", -88.02, 44.51),
    place("wi_madison", "Madison, WI", -89.40, 43.07),
    place("wv_huntington", "Huntington, WV", -82.45, 38.42),
    place("wv_morgantown", "Morgantown, WV", -79.96, 39.63),
    place("wy_casper", "Casper, WY", -106.31, 42.85),
    place("wy_jackson", "Jackson, WY", -110.76, 43.48),
];

pub const MICRO_US_PLACE_PRESETS: &[PlacePreset] = &[
    place("al_dothan", "Dothan, AL", -85.39, 31.22),
    place("ar_texarkana", "Texarkana, AR", -94.04, 33.43),
    place("az_flagstaff", "Flagstaff, AZ", -111.65, 35.20),
    place("az_yuma", "Yuma, AZ", -114.62, 32.69),
    place("ca_arcata", "Arcata, CA", -124.09, 40.87),
    place("ca_chico", "Chico, CA", -121.84, 39.73),
    place("ca_eureka", "Eureka, CA", -124.16, 40.80),
    place("ca_napa", "Napa, CA", -122.29, 38.30),
    place("ca_oxnard", "Oxnard, CA", -119.18, 34.20),
    place("ca_santa_cruz", "Santa Cruz, CA", -122.03, 36.97),
    place("ca_truckee", "Truckee, CA", -120.18, 39.33),
    place("ca_ukiah", "Ukiah, CA", -123.21, 39.15),
    place("ca_ventura", "Ventura, CA", -119.29, 34.27),
    place("co_durango", "Durango, CO", -107.88, 37.27),
    place(
        "co_steamboat_springs",
        "Steamboat Springs, CO",
        -106.83,
        40.49,
    ),
    place("fl_key_west", "Key West, FL", -81.78, 24.56),
    place("fl_panama_city", "Panama City, FL", -85.66, 30.16),
    place("ga_valdosta", "Valdosta, GA", -83.28, 30.83),
    place("id_lewiston", "Lewiston, ID", -117.02, 46.42),
    place("id_twin_falls", "Twin Falls, ID", -114.46, 42.56),
    place("il_quincy", "Quincy, IL", -91.40, 39.94),
    place("in_bloomington", "Bloomington, IN", -86.53, 39.17),
    place("ks_hays", "Hays, KS", -99.33, 38.88),
    place("ky_owensboro", "Owensboro, KY", -87.11, 37.77),
    place("la_alexandria", "Alexandria, LA", -92.45, 31.31),
    place("me_bar_harbor", "Bar Harbor, ME", -68.20, 44.39),
    place("mi_marquette", "Marquette, MI", -87.40, 46.54),
    place("mn_mankato", "Mankato, MN", -93.99, 44.16),
    place("mo_joplin", "Joplin, MO", -94.51, 37.08),
    place("mt_kalispell", "Kalispell, MT", -114.32, 48.20),
    place("nc_boone", "Boone, NC", -81.67, 36.22),
    place("nv_winnemucca", "Winnemucca, NV", -117.74, 40.97),
    place("nm_farmington", "Farmington, NM", -108.22, 36.73),
    place("ny_lake_placid", "Lake Placid, NY", -73.98, 44.28),
    place("ny_watertown", "Watertown, NY", -75.91, 43.97),
    place("oh_athens", "Athens, OH", -82.10, 39.33),
    place("ok_bartlesville", "Bartlesville, OK", -95.98, 36.75),
    place("or_astoria", "Astoria, OR", -123.83, 46.19),
    place("or_klamath_falls", "Klamath Falls, OR", -121.78, 42.22),
    place("pa_williamsport", "Williamsport, PA", -77.00, 41.24),
    place("sc_beaufort", "Beaufort, SC", -80.67, 32.43),
    place("sd_spearfish", "Spearfish, SD", -103.86, 44.49),
    place("tn_cookeville", "Cookeville, TN", -85.50, 36.16),
    place("tx_del_rio", "Del Rio, TX", -100.90, 29.36),
    place("tx_harlingen", "Harlingen, TX", -97.70, 26.19),
    place("ut_price", "Price, UT", -110.81, 39.60),
    place("va_danville", "Danville, VA", -79.40, 36.59),
    place("wa_walla_walla", "Walla Walla, WA", -118.34, 46.07),
    place("wi_rhinelander", "Rhinelander, WI", -89.41, 45.64),
    place("wy_laramie", "Laramie, WY", -105.59, 41.31),
];

pub fn major_us_city_places() -> &'static [PlacePreset] {
    MAJOR_US_CITY_PRESETS
}

pub fn aux_us_city_places() -> &'static [PlacePreset] {
    AUX_US_CITY_PRESETS
}

pub fn micro_us_place_presets() -> &'static [PlacePreset] {
    MICRO_US_PLACE_PRESETS
}

pub fn major_us_city_domains() -> Vec<DomainSpec> {
    MAJOR_US_CITY_PRESETS
        .iter()
        .copied()
        .map(PlacePreset::domain)
        .collect()
}

pub fn select_major_us_city_places(
    bounds: (f64, f64, f64, f64),
    options: PlaceSelectionOptions,
) -> Vec<SelectedPlace> {
    select_places_for_bounds(MAJOR_US_CITY_PRESETS, bounds, options)
}

pub fn select_major_us_city_domains(
    bounds: (f64, f64, f64, f64),
    options: PlaceSelectionOptions,
) -> Vec<DomainSpec> {
    select_major_us_city_places(bounds, options)
        .into_iter()
        .map(|place| place.domain())
        .collect()
}

pub fn apply_place_label_overlay(
    request: &mut MapRenderRequest,
    overlay: &PlaceLabelOverlay,
    domain: &DomainSpec,
    grid_lat_deg: &[f32],
    grid_lon_deg: &[f32],
    projection: Option<&GridProjection>,
) -> Result<(), Box<dyn std::error::Error>> {
    if request.projected_domain.is_none() {
        return Ok(());
    }

    let Some(plan) = place_label_plan_for_domain(domain) else {
        return Ok(());
    };

    let selected = select_places_for_label_plan(domain, plan, overlay);
    if selected.is_empty() {
        return Ok(());
    }

    let geographic_points = selected
        .iter()
        .map(|place| (place.center_lat, place.center_lon))
        .collect::<Vec<_>>();
    let projected =
        project_geographic_points(&geographic_points, grid_lat_deg, grid_lon_deg, projection)?;
    let center_lon = (domain.bounds.0 + domain.bounds.1) * 0.5;
    let center_lat = (domain.bounds.2 + domain.bounds.3) * 0.5;

    request
        .projected_place_labels
        .extend(
            selected
                .into_iter()
                .zip(projected.into_iter())
                .map(|(place, (x, y))| {
                    let is_anchor = plan
                        .anchor_slug
                        .map(|anchor| anchor == place.slug.as_str())
                        .unwrap_or(false);
                    let mut style = place_label_style(
                        plan.kind,
                        is_anchor,
                        place_catalog_tier_for_slug(place.slug.as_str()),
                    );
                    style.label_placement = interior_label_placement(
                        center_lon,
                        center_lat,
                        place.center_lon,
                        place.center_lat,
                    );
                    ProjectedPlaceLabel::new(x, y)
                        .with_label(display_label_for_domain(plan.kind, &place.label))
                        .with_style(style)
                }),
        );

    Ok(())
}

pub fn apply_major_place_labels(
    request: &mut MapRenderRequest,
    domain: &DomainSpec,
    grid_lat_deg: &[f32],
    grid_lon_deg: &[f32],
    projection: Option<&GridProjection>,
) -> Result<(), Box<dyn std::error::Error>> {
    apply_place_label_overlay(
        request,
        &PlaceLabelOverlay::major_us_cities(),
        domain,
        grid_lat_deg,
        grid_lon_deg,
        projection,
    )
}

pub fn default_place_label_overlay_for_domain(
    domain: &DomainSpec,
    density: PlaceLabelDensityTier,
) -> Option<PlaceLabelOverlay> {
    place_label_plan_for_domain(domain)
        .map(|_| PlaceLabelOverlay::major_us_cities().with_density(density))
}

pub fn default_major_place_label_overlay_for_domain(
    domain: &DomainSpec,
) -> Option<PlaceLabelOverlay> {
    default_place_label_overlay_for_domain(domain, PlaceLabelDensityTier::Major)
}

pub fn select_places_for_bounds(
    catalog: &[PlacePreset],
    bounds: (f64, f64, f64, f64),
    options: PlaceSelectionOptions,
) -> Vec<SelectedPlace> {
    if matches!(options.max_count, Some(0)) {
        return Vec::new();
    }

    let center_lon = (bounds.0 + bounds.1) * 0.5;
    let center_lat = (bounds.2 + bounds.3) * 0.5;
    let region_diag_km = haversine_km(bounds.2, bounds.0, bounds.3, bounds.1).max(1.0);
    let region_half_width_km =
        ((bounds.1 - bounds.0).abs() * center_lat.to_radians().cos().abs() * KM_PER_DEG_LAT * 0.5)
            .max(1.0);
    let region_half_height_km = ((bounds.3 - bounds.2).abs() * KM_PER_DEG_LAT * 0.5).max(1.0);
    let max_source_index = catalog.len().saturating_sub(1).max(1) as f64;

    let mut candidates = catalog
        .iter()
        .enumerate()
        .filter_map(|(index, preset)| {
            let selection_bounds = effective_place_bounds(*preset);
            if !matches_place_bounds(*preset, selection_bounds, bounds, options.containment) {
                return None;
            }

            let center_distance_km =
                haversine_km(center_lat, center_lon, preset.center_lat, preset.center_lon);
            let edge_margin_km = edge_margin_km(*preset, bounds);
            let centrality_score = (1.0 - center_distance_km / region_diag_km).clamp(0.0, 1.0);
            let interior_score =
                (edge_margin_km / region_half_width_km.min(region_half_height_km)).clamp(0.0, 1.0);
            let source_score = 1.0 - index as f64 / max_source_index;
            let ranking_score =
                0.55 * interior_score + 0.35 * centrality_score + 0.10 * source_score;

            Some(RankedPlaceCandidate {
                place: SelectedPlace {
                    slug: preset.slug.to_string(),
                    label: preset.label.to_string(),
                    center_lon: preset.center_lon,
                    center_lat: preset.center_lat,
                    bounds: preset.bounds(),
                    source_index: index,
                    center_distance_km,
                    edge_margin_km,
                    ranking_score,
                },
                selection_bounds,
            })
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|left, right| {
        right
            .place
            .ranking_score
            .total_cmp(&left.place.ranking_score)
            .then_with(|| {
                left.place
                    .center_distance_km
                    .total_cmp(&right.place.center_distance_km)
            })
            .then_with(|| left.place.source_index.cmp(&right.place.source_index))
    });

    let mut selected = Vec::<RankedPlaceCandidate>::new();
    for candidate in candidates {
        let is_decluttered = selected.iter().all(|kept| {
            let center_spacing_ok = haversine_km(
                candidate.place.center_lat,
                candidate.place.center_lon,
                kept.place.center_lat,
                kept.place.center_lon,
            ) >= options.min_center_spacing_km.max(0.0);
            let crop_overlap_ok =
                crop_overlap_fraction(candidate.selection_bounds, kept.selection_bounds)
                    <= options.max_crop_overlap_fraction.clamp(0.0, 1.0);
            center_spacing_ok && crop_overlap_ok
        });
        if !is_decluttered {
            continue;
        }
        selected.push(candidate);
        if let Some(limit) = options.max_count {
            if selected.len() >= limit {
                break;
            }
        }
    }

    selected
        .into_iter()
        .map(|candidate| candidate.place)
        .collect()
}

pub fn centered_domain<S: Into<String>>(
    slug: S,
    center_lon: f64,
    center_lat: f64,
    half_height_deg: f64,
) -> DomainSpec {
    DomainSpec::new(
        slug,
        centered_bounds(
            center_lon,
            center_lat,
            half_height_deg,
            PLACE_OUTPUT_ASPECT_RATIO,
        ),
    )
}

pub fn centered_bounds(
    center_lon: f64,
    center_lat: f64,
    half_height_deg: f64,
    aspect_ratio: f64,
) -> (f64, f64, f64, f64) {
    let cos_lat = center_lat.to_radians().cos().abs().max(0.25);
    let half_width_deg = half_height_deg * aspect_ratio / cos_lat;
    (
        center_lon - half_width_deg,
        center_lon + half_width_deg,
        center_lat - half_height_deg,
        center_lat + half_height_deg,
    )
}

fn place_label_plan_for_domain(domain: &DomainSpec) -> Option<PlaceLabelPlan> {
    if domain.slug == "conus" {
        return Some(PlaceLabelPlan {
            kind: PlaceLabelDomainKind::Conus,
            max_count: 16,
            min_center_spacing_km: 360.0,
            max_crop_overlap_fraction: 0.20,
            anchor_slug: None,
        });
    }

    if REGION_PLACE_LABEL_SLUGS
        .iter()
        .copied()
        .any(|slug| slug == domain.slug.as_str())
    {
        let (max_count, min_center_spacing_km) = match domain.slug.as_str() {
            "california" | "california_square" | "california_southwest" => (7, 150.0),
            "reno_square" => (4, 110.0),
            _ => (8, 220.0),
        };
        return Some(PlaceLabelPlan {
            kind: PlaceLabelDomainKind::Region,
            max_count,
            min_center_spacing_km,
            max_crop_overlap_fraction: 0.35,
            anchor_slug: None,
        });
    }

    major_us_city_places()
        .iter()
        .find(|preset| preset.slug == domain.slug.as_str())
        .map(|preset| PlaceLabelPlan {
            kind: PlaceLabelDomainKind::CityCrop,
            max_count: 4,
            min_center_spacing_km: 55.0,
            max_crop_overlap_fraction: 0.80,
            anchor_slug: Some(preset.slug),
        })
}

fn select_places_for_label_plan(
    domain: &DomainSpec,
    plan: PlaceLabelPlan,
    overlay: &PlaceLabelOverlay,
) -> Vec<SelectedPlace> {
    let Some(plan) = tuned_place_label_plan(plan, overlay.density) else {
        return Vec::new();
    };

    let options = match plan.kind {
        PlaceLabelDomainKind::Conus | PlaceLabelDomainKind::Region => {
            PlaceSelectionOptions::for_overlay_labels()
                .with_max_count(plan.max_count)
                .with_min_center_spacing_km(plan.min_center_spacing_km)
                .with_max_crop_overlap_fraction(plan.max_crop_overlap_fraction)
        }
        PlaceLabelDomainKind::CityCrop => PlaceSelectionOptions::for_city_crops()
            .with_max_count(plan.max_count)
            .with_min_center_spacing_km(plan.min_center_spacing_km)
            .with_max_crop_overlap_fraction(plan.max_crop_overlap_fraction),
    };

    let catalog = overlay.filtered_catalog();
    let mut selected = select_places_for_bounds(&catalog, domain.bounds, options);
    if let Some(anchor_slug) = plan.anchor_slug {
        if let Some(anchor) = catalog
            .iter()
            .copied()
            .find(|preset| preset.slug == anchor_slug)
            .filter(|preset| contains_point(domain.bounds, preset.center_lon, preset.center_lat))
        {
            if !selected.iter().any(|place| place.slug == anchor_slug) {
                selected.insert(
                    0,
                    SelectedPlace {
                        slug: anchor.slug.to_string(),
                        label: anchor.label.to_string(),
                        center_lon: anchor.center_lon,
                        center_lat: anchor.center_lat,
                        bounds: anchor.bounds(),
                        source_index: 0,
                        center_distance_km: 0.0,
                        edge_margin_km: edge_margin_km(anchor, domain.bounds),
                        ranking_score: f64::INFINITY,
                    },
                );
            }
        }

        selected.sort_by(|left, right| {
            let left_anchor = left.slug == anchor_slug;
            let right_anchor = right.slug == anchor_slug;
            right_anchor
                .cmp(&left_anchor)
                .then_with(|| right.ranking_score.total_cmp(&left.ranking_score))
                .then_with(|| left.center_distance_km.total_cmp(&right.center_distance_km))
        });
        selected.truncate(plan.max_count);
    }

    selected
}

fn tuned_place_label_plan(
    mut plan: PlaceLabelPlan,
    density: PlaceLabelDensityTier,
) -> Option<PlaceLabelPlan> {
    match density {
        PlaceLabelDensityTier::None => return None,
        PlaceLabelDensityTier::Major => {}
        PlaceLabelDensityTier::MajorAndAux => {
            plan.max_count = plan.max_count.saturating_mul(2);
            plan.min_center_spacing_km = (plan.min_center_spacing_km * 0.70).max(35.0);
            plan.max_crop_overlap_fraction = (plan.max_crop_overlap_fraction + 0.18).min(0.90);
        }
        PlaceLabelDensityTier::Dense => {
            plan.max_count = plan.max_count.saturating_mul(4);
            plan.min_center_spacing_km = (plan.min_center_spacing_km * 0.45).max(20.0);
            plan.max_crop_overlap_fraction = (plan.max_crop_overlap_fraction + 0.40).min(0.97);
        }
    }
    Some(plan)
}

fn place_catalog_tier_for_slug(slug: &str) -> PlaceCatalogTier {
    if MAJOR_US_CITY_PRESETS
        .iter()
        .any(|preset| preset.slug == slug)
    {
        PlaceCatalogTier::Major
    } else if AUX_US_CITY_PRESETS.iter().any(|preset| preset.slug == slug) {
        PlaceCatalogTier::Aux
    } else if MICRO_US_PLACE_PRESETS
        .iter()
        .any(|preset| preset.slug == slug)
    {
        PlaceCatalogTier::Micro
    } else {
        PlaceCatalogTier::Major
    }
}

fn effective_place_bounds(preset: PlacePreset) -> (f64, f64, f64, f64) {
    centered_bounds(
        preset.center_lon,
        preset.center_lat,
        effective_place_half_height_deg(preset),
        PLACE_OUTPUT_ASPECT_RATIO,
    )
}

fn effective_place_half_height_deg(preset: PlacePreset) -> f64 {
    match place_catalog_tier_for_slug(preset.slug) {
        PlaceCatalogTier::Major => preset.half_height_deg,
        PlaceCatalogTier::Aux => AUX_PLACE_HALF_HEIGHT_DEG.min(preset.half_height_deg),
        PlaceCatalogTier::Micro => MICRO_PLACE_HALF_HEIGHT_DEG.min(preset.half_height_deg),
    }
}

fn place_label_style(
    kind: PlaceLabelDomainKind,
    is_anchor: bool,
    catalog_tier: PlaceCatalogTier,
) -> ProjectedPlaceLabelStyle {
    match (kind, is_anchor) {
        (PlaceLabelDomainKind::CityCrop, true) => ProjectedPlaceLabelStyle {
            marker_radius_px: 4,
            marker_fill: Color::rgba(255, 255, 255, 240),
            marker_outline: Color::rgba(23, 29, 37, 255),
            marker_outline_width: 2,
            label_color: Color::rgba(23, 29, 37, 255),
            label_halo: Color::rgba(255, 255, 255, 240),
            label_halo_width_px: 2,
            label_scale: 2,
            label_offset_x_px: 8,
            label_offset_y_px: -4,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: true,
        },
        (PlaceLabelDomainKind::CityCrop, false) => city_crop_place_style(catalog_tier),
        (PlaceLabelDomainKind::Conus, _) => {
            broad_domain_place_style(PlaceLabelDomainKind::Conus, catalog_tier)
        }
        (PlaceLabelDomainKind::Region, _) => {
            broad_domain_place_style(PlaceLabelDomainKind::Region, catalog_tier)
        }
    }
}

fn city_crop_place_style(catalog_tier: PlaceCatalogTier) -> ProjectedPlaceLabelStyle {
    match catalog_tier {
        PlaceCatalogTier::Major => ProjectedPlaceLabelStyle {
            marker_radius_px: 3,
            marker_fill: Color::rgba(255, 255, 255, 230),
            marker_outline: Color::rgba(35, 44, 55, 235),
            marker_outline_width: 1,
            label_color: Color::rgba(28, 34, 42, 255),
            label_halo: Color::rgba(255, 255, 255, 230),
            label_halo_width_px: 2,
            label_scale: 1,
            label_offset_x_px: 7,
            label_offset_y_px: -3,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: false,
        },
        PlaceCatalogTier::Aux => ProjectedPlaceLabelStyle {
            marker_radius_px: 2,
            marker_fill: Color::rgba(255, 255, 255, 205),
            marker_outline: Color::rgba(44, 54, 66, 205),
            marker_outline_width: 1,
            label_color: Color::rgba(42, 50, 61, 230),
            label_halo: Color::rgba(255, 255, 255, 210),
            label_halo_width_px: 2,
            label_scale: 1,
            label_offset_x_px: 5,
            label_offset_y_px: -2,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: false,
        },
        PlaceCatalogTier::Micro => ProjectedPlaceLabelStyle {
            marker_radius_px: 1,
            marker_fill: Color::rgba(255, 255, 255, 180),
            marker_outline: Color::rgba(0, 0, 0, 0),
            marker_outline_width: 0,
            label_color: Color::rgba(66, 74, 84, 205),
            label_halo: Color::rgba(255, 255, 255, 185),
            label_halo_width_px: 1,
            label_scale: 1,
            label_offset_x_px: 4,
            label_offset_y_px: -1,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: false,
        },
    }
}

fn broad_domain_place_style(
    kind: PlaceLabelDomainKind,
    catalog_tier: PlaceCatalogTier,
) -> ProjectedPlaceLabelStyle {
    match (kind, catalog_tier) {
        (PlaceLabelDomainKind::Conus, PlaceCatalogTier::Major) => ProjectedPlaceLabelStyle {
            marker_radius_px: 2,
            marker_fill: Color::rgba(255, 255, 255, 220),
            marker_outline: Color::rgba(38, 48, 60, 220),
            marker_outline_width: 1,
            label_color: Color::rgba(28, 34, 42, 245),
            label_halo: Color::rgba(255, 255, 255, 220),
            label_halo_width_px: 2,
            label_scale: 1,
            label_offset_x_px: 6,
            label_offset_y_px: -2,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: false,
        },
        (PlaceLabelDomainKind::Conus, PlaceCatalogTier::Aux) => ProjectedPlaceLabelStyle {
            marker_radius_px: 1,
            marker_fill: Color::rgba(255, 255, 255, 185),
            marker_outline: Color::rgba(40, 49, 61, 185),
            marker_outline_width: 1,
            label_color: Color::rgba(58, 67, 77, 215),
            label_halo: Color::rgba(255, 255, 255, 190),
            label_halo_width_px: 1,
            label_scale: 1,
            label_offset_x_px: 4,
            label_offset_y_px: -1,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: false,
        },
        (PlaceLabelDomainKind::Conus, PlaceCatalogTier::Micro) => ProjectedPlaceLabelStyle {
            marker_radius_px: 0,
            marker_fill: Color::rgba(0, 0, 0, 0),
            marker_outline: Color::rgba(0, 0, 0, 0),
            marker_outline_width: 0,
            label_color: Color::rgba(82, 92, 103, 180),
            label_halo: Color::rgba(255, 255, 255, 170),
            label_halo_width_px: 1,
            label_scale: 1,
            label_offset_x_px: 2,
            label_offset_y_px: 0,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: false,
        },
        (PlaceLabelDomainKind::Region, PlaceCatalogTier::Major) => ProjectedPlaceLabelStyle {
            marker_radius_px: 3,
            marker_fill: Color::rgba(255, 255, 255, 230),
            marker_outline: Color::rgba(28, 34, 42, 235),
            marker_outline_width: 1,
            label_color: Color::rgba(24, 31, 39, 250),
            label_halo: Color::rgba(255, 255, 255, 230),
            label_halo_width_px: 2,
            label_scale: 1,
            label_offset_x_px: 6,
            label_offset_y_px: -2,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: false,
        },
        (PlaceLabelDomainKind::Region, PlaceCatalogTier::Aux) => ProjectedPlaceLabelStyle {
            marker_radius_px: 2,
            marker_fill: Color::rgba(255, 255, 255, 200),
            marker_outline: Color::rgba(35, 42, 51, 200),
            marker_outline_width: 1,
            label_color: Color::rgba(50, 59, 69, 225),
            label_halo: Color::rgba(255, 255, 255, 205),
            label_halo_width_px: 1,
            label_scale: 1,
            label_offset_x_px: 5,
            label_offset_y_px: -1,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: false,
        },
        (PlaceLabelDomainKind::Region, PlaceCatalogTier::Micro) => ProjectedPlaceLabelStyle {
            marker_radius_px: 1,
            marker_fill: Color::rgba(255, 255, 255, 165),
            marker_outline: Color::rgba(0, 0, 0, 0),
            marker_outline_width: 0,
            label_color: Color::rgba(74, 83, 94, 200),
            label_halo: Color::rgba(255, 255, 255, 170),
            label_halo_width_px: 1,
            label_scale: 1,
            label_offset_x_px: 4,
            label_offset_y_px: 0,
            label_placement: ProjectedLabelPlacement::AboveRight,
            label_bold: false,
        },
        (PlaceLabelDomainKind::CityCrop, _) => unreachable!(),
    }
}

fn display_label_for_domain(kind: PlaceLabelDomainKind, label: &str) -> String {
    match kind {
        PlaceLabelDomainKind::Conus => label.to_string(),
        PlaceLabelDomainKind::Region | PlaceLabelDomainKind::CityCrop => compact_place_label(label),
    }
}

fn compact_place_label(label: &str) -> String {
    label
        .split_once(',')
        .map(|(place, _)| place.trim().to_string())
        .unwrap_or_else(|| label.to_string())
}

fn interior_label_placement(
    center_lon: f64,
    center_lat: f64,
    place_lon: f64,
    place_lat: f64,
) -> ProjectedLabelPlacement {
    let horizontal_right = place_lon <= center_lon;
    let vertical_above = place_lat <= center_lat;
    match (vertical_above, horizontal_right) {
        (true, true) => ProjectedLabelPlacement::AboveRight,
        (true, false) => ProjectedLabelPlacement::AboveLeft,
        (false, true) => ProjectedLabelPlacement::BelowRight,
        (false, false) => ProjectedLabelPlacement::BelowLeft,
    }
}

const fn place(
    slug: &'static str,
    label: &'static str,
    center_lon: f64,
    center_lat: f64,
) -> PlacePreset {
    PlacePreset {
        slug,
        label,
        center_lon,
        center_lat,
        half_height_deg: DEFAULT_PLACE_HALF_HEIGHT_DEG,
    }
}

fn matches_place_bounds(
    preset: PlacePreset,
    crop_bounds: (f64, f64, f64, f64),
    query_bounds: (f64, f64, f64, f64),
    containment: PlaceContainmentMode,
) -> bool {
    match containment {
        PlaceContainmentMode::CenterWithinBounds => {
            contains_point(query_bounds, preset.center_lon, preset.center_lat)
        }
        PlaceContainmentMode::CropIntersectsBounds => bounds_intersect(crop_bounds, query_bounds),
        PlaceContainmentMode::CropWithinBounds => bounds_within(crop_bounds, query_bounds),
    }
}

fn contains_point(bounds: (f64, f64, f64, f64), lon: f64, lat: f64) -> bool {
    lon >= bounds.0 && lon <= bounds.1 && lat >= bounds.2 && lat <= bounds.3
}

fn bounds_intersect(left: (f64, f64, f64, f64), right: (f64, f64, f64, f64)) -> bool {
    left.0 <= right.1 && left.1 >= right.0 && left.2 <= right.3 && left.3 >= right.2
}

fn bounds_within(inner: (f64, f64, f64, f64), outer: (f64, f64, f64, f64)) -> bool {
    inner.0 >= outer.0 && inner.1 <= outer.1 && inner.2 >= outer.2 && inner.3 <= outer.3
}

fn edge_margin_km(preset: PlacePreset, bounds: (f64, f64, f64, f64)) -> f64 {
    let cos_lat = preset.center_lat.to_radians().cos().abs().max(0.25);
    let west_margin = (preset.center_lon - bounds.0).abs() * cos_lat * KM_PER_DEG_LAT;
    let east_margin = (bounds.1 - preset.center_lon).abs() * cos_lat * KM_PER_DEG_LAT;
    let south_margin = (preset.center_lat - bounds.2).abs() * KM_PER_DEG_LAT;
    let north_margin = (bounds.3 - preset.center_lat).abs() * KM_PER_DEG_LAT;
    west_margin
        .min(east_margin)
        .min(south_margin)
        .min(north_margin)
}

fn crop_overlap_fraction(left: (f64, f64, f64, f64), right: (f64, f64, f64, f64)) -> f64 {
    let west = left.0.max(right.0);
    let east = left.1.min(right.1);
    let south = left.2.max(right.2);
    let north = left.3.min(right.3);
    if east <= west || north <= south {
        return 0.0;
    }

    let overlap = bounds_area_km2((west, east, south, north));
    if overlap <= 0.0 {
        return 0.0;
    }
    let baseline = bounds_area_km2(left).min(bounds_area_km2(right)).max(1.0);
    (overlap / baseline).clamp(0.0, 1.0)
}

fn bounds_area_km2(bounds: (f64, f64, f64, f64)) -> f64 {
    let center_lat = (bounds.2 + bounds.3) * 0.5;
    let width_km = (bounds.1 - bounds.0).abs()
        * center_lat.to_radians().cos().abs().max(0.25)
        * KM_PER_DEG_LAT;
    let height_km = (bounds.3 - bounds.2).abs() * KM_PER_DEG_LAT;
    width_km * height_km
}

fn haversine_km(lat0_deg: f64, lon0_deg: f64, lat1_deg: f64, lon1_deg: f64) -> f64 {
    let lat0 = lat0_deg.to_radians();
    let lat1 = lat1_deg.to_radians();
    let dlat = (lat1_deg - lat0_deg).to_radians();
    let dlon = (lon1_deg - lon0_deg).to_radians();
    let a = (dlat * 0.5).sin().powi(2) + lat0.cos() * lat1.cos() * (dlon * 0.5).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    6371.0 * c
}

fn project_geographic_points(
    geographic_points: &[(f64, f64)],
    grid_lat_deg: &[f32],
    grid_lon_deg: &[f32],
    projection: Option<&GridProjection>,
) -> Result<Vec<(f64, f64)>, Box<dyn std::error::Error>> {
    if geographic_points.is_empty() {
        return Ok(Vec::new());
    }
    let lat = geographic_points
        .iter()
        .map(|&(lat_deg, _)| lat_deg as f32)
        .collect::<Vec<_>>();
    let lon = geographic_points
        .iter()
        .map(|&(_, lon_deg)| lon_deg as f32)
        .collect::<Vec<_>>();

    let mut options = ProjectedDomainBuildOptions::full_domain(1.0);
    if let Some(reference_latitude_deg) = latitude_midpoint_deg(grid_lat_deg) {
        options = options.with_reference_latitude(reference_latitude_deg);
    }
    if let Some(projection_spec) = resolve_projection_spec(grid_lat_deg, grid_lon_deg, projection) {
        options = options.with_projection(projection_spec);
    }
    let projected = build_projected_domain(&lat, &lon, &options)?;
    Ok(projected.x.into_iter().zip(projected.y).collect())
}

fn resolve_projection_spec(
    grid_lat_deg: &[f32],
    grid_lon_deg: &[f32],
    projection: Option<&GridProjection>,
) -> Option<ProjectionSpec> {
    projection
        .cloned()
        .map(Into::into)
        .or_else(|| ProjectionSpec::infer_from_latlon_grid(grid_lat_deg, grid_lon_deg))
}

fn latitude_midpoint_deg(values: &[f32]) -> Option<f64> {
    let mut min_lat = f64::INFINITY;
    let mut max_lat = f64::NEG_INFINITY;
    for &value in values {
        let value = value as f64;
        if !value.is_finite() {
            continue;
        }
        min_lat = min_lat.min(value);
        max_lat = max_lat.max(value);
    }
    if min_lat.is_finite() && max_lat.is_finite() {
        Some((min_lat + max_lat) * 0.5)
    } else {
        None
    }
}

#[cfg(test)]
mod tests;
