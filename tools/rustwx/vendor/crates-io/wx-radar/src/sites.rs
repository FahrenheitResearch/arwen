//! NEXRAD radar site database.
//!
//! Provides a static table of operational NEXRAD (WSR-88D) sites and a lookup
//! function by site identifier.
//!
//! Research/test radars (KOUN, KCRI) are excluded — use `--site` directly if needed.

use wx_field::RadarSite;

/// Operational NEXRAD sites with coordinates and elevation.
///
/// Coordinates sourced from NWS NEXRAD site documentation and gr2rust site list.
/// Elevations are provided where known; 0.0 is used as a placeholder otherwise.
pub static SITES: &[(&str, &str, f64, f64, f64)] = &[
    ("KABR", "Aberdeen, SD",                45.4558, -98.4131,    0.0),
    ("KABX", "Albuquerque, NM",             35.1497, -106.824,    0.0),
    ("KAKQ", "Wakefield, VA",               36.9839, -77.0075,    0.0),
    ("KAMA", "Amarillo, TX",                35.2333, -101.709,    0.0),
    ("KAMX", "Miami, FL",                   25.6111, -80.4128,    4.0),
    ("KAPX", "Gaylord, MI",                 44.9072, -84.7197,    0.0),
    ("KARX", "La Crosse, WI",               43.8228, -91.1911,    0.0),
    ("KATX", "Seattle, WA",                 48.1944, -122.496,  151.0),
    ("KBBX", "Beale AFB, CA",               39.4961, -121.632,    0.0),
    ("KBGM", "Binghamton, NY",              42.1997, -75.985,     0.0),
    ("KBHX", "Eureka, CA",                  40.4986, -124.292,    0.0),
    ("KBIS", "Bismarck, ND",                46.7708, -100.76,     0.0),
    ("KBLX", "Billings, MT",                45.8536, -108.607,    0.0),
    ("KBMX", "Birmingham, AL",              33.1722, -86.7697,    0.0),
    ("KBOX", "Boston, MA",                  41.9558, -71.1369,    0.0),
    ("KBRO", "Brownsville, TX",             25.9161, -97.4189,    0.0),
    ("KBUF", "Buffalo, NY",                 42.9486, -78.7369,    0.0),
    ("KBYX", "Key West, FL",                24.5975, -81.7031,    0.0),
    ("KCAE", "Columbia, SC",                33.9486, -81.1186,    0.0),
    ("KCBW", "Caribou, ME",                 46.0392, -67.8064,    0.0),
    ("KCBX", "Boise, ID",                   43.4908, -116.236,    0.0),
    ("KCCX", "State College, PA",            40.9228, -78.0039,    0.0),
    ("KCLE", "Cleveland, OH",               41.4131, -81.86,      0.0),
    ("KCLX", "Charleston, SC",              32.6556, -81.0422,    0.0),
    ("KCRP", "Corpus Christi, TX",          27.7842, -97.5111,    0.0),
    ("KCXX", "Burlington, VT",              44.5111, -73.1667,    0.0),
    ("KCYS", "Cheyenne, WY",                41.1519, -104.806,    0.0),
    ("KDAX", "Sacramento, CA",              38.5011, -121.678,    0.0),
    ("KDDC", "Dodge City, KS",              37.7608, -99.9689,    0.0),
    ("KDFX", "Laughlin AFB, TX",            29.2725, -100.281,    0.0),
    ("KDGX", "Brandon, MS",                 32.28,   -89.9844,    0.0),
    ("KDIX", "Philadelphia, NJ",            39.9469, -74.4108,    0.0),
    ("KDLH", "Duluth, MN",                  46.8369, -92.2097,    0.0),
    ("KDMX", "Des Moines, IA",              41.7311, -93.7228,    0.0),
    ("KDOX", "Dover AFB, DE",               38.8256, -75.44,      0.0),
    ("KDTX", "Detroit, MI",                 42.6997, -83.4717,    0.0),
    ("KDVN", "Davenport, IA",               41.6117, -90.5808,    0.0),
    ("KDYX", "Dyess AFB, TX",               32.5386, -99.2542,    0.0),
    ("KEAX", "Kansas City, MO",             38.8103, -94.2644,    0.0),
    ("KEMX", "Tucson, AZ",                  31.8936, -110.63,     0.0),
    ("KENX", "Albany, NY",                  42.5864, -74.0639,    0.0),
    ("KEOX", "Fort Rucker, AL",             31.4606, -85.4594,    0.0),
    ("KEPZ", "El Paso, TX",                 31.8731, -106.698,    0.0),
    ("KESX", "Las Vegas, NV",               35.7011, -114.892,    0.0),
    ("KEVX", "NW Florida, FL",              30.5644, -85.9214,    0.0),
    ("KEWX", "Austin/San Antonio, TX",      29.7039, -98.0286,    0.0),
    ("KEYX", "Edwards AFB, CA",             35.0978, -117.561,    0.0),
    ("KFCX", "Roanoke, VA",                 37.0242, -80.2742,    0.0),
    ("KFDR", "Frederick, OK",               34.3622, -98.9764,    0.0),
    ("KFDX", "Cannon AFB, NM",              34.6353, -103.631,    0.0),
    ("KFFC", "Atlanta, GA",                 33.3636, -84.5658,  262.0),
    ("KFSD", "Sioux Falls, SD",             43.5878, -96.7292,    0.0),
    ("KFSX", "Flagstaff, AZ",               34.5744, -111.198,    0.0),
    ("KFTG", "Denver, CO",                  39.7867, -104.546,    0.0),
    ("KFWS", "Dallas/Fort Worth, TX",       32.5731, -97.3028,  208.0),
    ("KGGW", "Glasgow, MT",                 48.2064, -106.625,    0.0),
    ("KGJX", "Grand Junction, CO",          39.0619, -108.214,    0.0),
    ("KGLD", "Goodland, KS",                39.3667, -101.7,      0.0),
    ("KGRB", "Green Bay, WI",               44.4986, -88.1111,    0.0),
    ("KGRK", "Central Texas, TX",           30.7217, -97.3828,    0.0),
    ("KGRR", "Grand Rapids, MI",            42.8939, -85.5447,    0.0),
    ("KGSP", "Greenville, SC",              34.8833, -82.22,      0.0),
    ("KGWX", "Columbus AFB, MS",            33.8967, -88.3289,    0.0),
    ("KGYX", "Portland, ME",                43.8914, -70.2564,    0.0),
    ("KHDX", "Holloman AFB, NM",            33.0769, -106.123,    0.0),
    ("KHGX", "Houston, TX",                 29.4719, -95.0792,    0.0),
    ("KHNX", "Hanford, CA",                 36.3142, -119.632,    0.0),
    ("KHPX", "Fort Campbell, KY",           36.7369, -87.285,     0.0),
    ("KHTX", "Huntsville, AL",              34.9306, -86.0833,    0.0),
    ("KICT", "Wichita, KS",                 37.6544, -97.4428,    0.0),
    ("KICX", "Cedar City, UT",              37.5908, -112.862,    0.0),
    ("KILN", "Cincinnati, OH",              39.4203, -83.8217,    0.0),
    ("KILX", "Lincoln, IL",                 40.1506, -89.3369,    0.0),
    ("KIND", "Indianapolis, IN",            39.7075, -86.2803,    0.0),
    ("KINX", "Tulsa, OK",                   36.175,  -95.5644,    0.0),
    ("KIWA", "Phoenix, AZ",                 33.2892, -111.67,     0.0),
    ("KIWX", "N Indiana, IN",               41.3586, -85.7,       0.0),
    ("KJAX", "Jacksonville, FL",            30.4847, -81.7019,    0.0),
    ("KJGX", "Robins AFB, GA",              32.675,  -83.3511,    0.0),
    ("KJKL", "Jackson, KY",                 37.5908, -83.3131,    0.0),
    ("KLBB", "Lubbock, TX",                 33.6542, -101.814,    0.0),
    ("KLCH", "Lake Charles, LA",            30.125,  -93.2158,    0.0),
    ("KLIX", "New Orleans, LA",             30.3367, -89.8256,    8.0),
    ("KLNX", "North Platte, NE",            41.9578, -100.576,    0.0),
    ("KLOT", "Chicago, IL",                 41.6044, -88.0847,  202.0),
    ("KLRX", "Elko, NV",                    40.7397, -116.803,    0.0),
    ("KLSX", "St. Louis, MO",               38.6986, -90.6828,  185.0),
    ("KLTX", "Wilmington, NC",              33.9892, -78.4292,    0.0),
    ("KLVX", "Louisville, KY",              37.975,  -85.9436,    0.0),
    ("KLWX", "Sterling, VA",                38.9753, -77.4778,    0.0),
    ("KLZK", "Little Rock, AR",             34.8364, -92.2622,    0.0),
    ("KMAF", "Midland/Odessa, TX",          31.9433, -102.189,    0.0),
    ("KMAX", "Medford, OR",                 42.0811, -122.717,    0.0),
    ("KMBX", "Minot AFB, ND",               48.3925, -100.864,    0.0),
    ("KMHX", "Morehead City, NC",           34.7761, -76.8761,    0.0),
    ("KMKX", "Milwaukee, WI",               42.9678, -88.5506,    0.0),
    ("KMLB", "Melbourne, FL",               28.1133, -80.6542,    0.0),
    ("KMOB", "Mobile, AL",                  30.6794, -88.2397,    0.0),
    ("KMPX", "Minneapolis, MN",             44.8489, -93.5653,    0.0),
    ("KMQT", "Marquette, MI",               46.5311, -87.5486,    0.0),
    ("KMRX", "Knoxville, TN",               36.1686, -83.4017,    0.0),
    ("KMSX", "Missoula, MT",                47.0411, -113.986,    0.0),
    ("KMTX", "Salt Lake City, UT",          41.2628, -112.448,    0.0),
    ("KMUX", "San Francisco, CA",           37.1553, -121.898,    0.0),
    ("KMVX", "Fargo, ND",                   47.5278, -97.325,     0.0),
    ("KNKX", "San Diego, CA",               32.9189, -117.042,    0.0),
    ("KNQA", "Memphis, TN",                 35.3447, -89.8733,    0.0),
    ("KOAX", "Omaha, NE",                   41.3203, -96.3669,    0.0),
    ("KOHX", "Nashville, TN",               36.2472, -86.5625,    0.0),
    ("KOKX", "New York City, NY",           40.8656, -72.8639,   26.0),
    ("KOTX", "Spokane, WA",                 47.6806, -117.627,    0.0),
    ("KPAH", "Paducah, KY",                 37.0683, -88.7719,    0.0),
    ("KPBZ", "Pittsburgh, PA",              40.5317, -80.0178,    0.0),
    ("KPDT", "Pendleton, OR",               45.6906, -118.853,    0.0),
    ("KPOE", "Fort Polk, LA",               31.1556, -92.9758,    0.0),
    ("KPUX", "Pueblo, CO",                  38.4594, -104.181, 1600.0),
    ("KRAX", "Raleigh, NC",                 35.6653, -78.49,      0.0),
    ("KRGX", "Reno, NV",                    39.7542, -119.462,    0.0),
    ("KRIW", "Riverton, WY",                43.0661, -108.477,    0.0),
    ("KRLX", "Charleston, WV",              38.3111, -81.7228,    0.0),
    ("KRTX", "Portland, OR",                45.715,  -122.965,    0.0),
    ("KSFX", "Pocatello, ID",               43.1058, -112.686,    0.0),
    ("KSGF", "Springfield, MO",             37.235,  -93.4006,    0.0),
    ("KSHV", "Shreveport, LA",              32.4508, -93.8414,    0.0),
    ("KSJT", "San Angelo, TX",              31.3711, -100.493,    0.0),
    ("KSOX", "Santa Ana Mtns, CA",          33.8178, -117.636,  923.0),
    ("KSRX", "Fort Smith, AR",              35.2906, -94.3619,    0.0),
    ("KTBW", "Tampa Bay, FL",               27.7056, -82.4017,    0.0),
    ("KTFX", "Great Falls, MT",             47.4597, -111.385,    0.0),
    ("KTLH", "Tallahassee, FL",             30.3975, -84.3289,    0.0),
    ("KTLX", "Oklahoma City, OK",           35.3331, -97.2778,  370.0),
    ("KTWX", "Topeka, KS",                  38.9969, -96.2325,    0.0),
    ("KTYX", "Montague, NY",                43.7556, -75.68,      0.0),
    ("KUDX", "Rapid City, SD",              44.125,  -102.83,     0.0),
    ("KUEX", "Hastings, NE",                40.3208, -98.4419,    0.0),
    ("KVAX", "Moody AFB, GA",               30.8903, -83.0019,    0.0),
    ("KVBX", "Vandenberg AFB, CA",          34.8383, -120.398,    0.0),
    ("KVNX", "Vance AFB, OK",               36.7408, -98.1275,    0.0),
    ("KVTX", "Los Angeles, CA",             34.4117, -119.179,    0.0),
    ("KVWX", "Evansville, IN",              38.2603, -87.7247,    0.0),
    ("KYUX", "Yuma, AZ",                    32.4953, -114.657,    0.0),
];

/// Look up a NEXRAD site by its identifier (case-insensitive).
///
/// Returns `Some(RadarSite)` if the site is found in the built-in table,
/// or `None` otherwise.
///
/// # Example
/// ```
/// use wx_radar::sites::find_site;
/// let site = find_site("KTLX").unwrap();
/// assert_eq!(site.name, "Oklahoma City, OK");
/// ```
pub fn find_site(id: &str) -> Option<RadarSite> {
    let id_upper = id.to_uppercase();
    SITES.iter().find(|(sid, _, _, _, _)| *sid == id_upper).map(
        |(sid, name, lat, lon, elev)| RadarSite::new(*sid, *name, *lat, *lon, *elev),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_site_exists() {
        let site = find_site("KTLX").expect("KTLX should be in the site table");
        assert_eq!(site.id, "KTLX");
        assert_eq!(site.name, "Oklahoma City, OK");
        assert!((site.lat - 35.3331).abs() < 0.001);
        assert!((site.lon - (-97.2778)).abs() < 0.001);
    }

    #[test]
    fn test_find_site_case_insensitive() {
        assert!(find_site("ktlx").is_some());
        assert!(find_site("Ktlx").is_some());
    }

    #[test]
    fn test_find_site_not_found() {
        assert!(find_site("XYZZ").is_none());
    }

    #[test]
    fn test_sites_table_has_minimum_entries() {
        assert!(SITES.len() >= 140, "Expected 140+ sites, got {}", SITES.len());
    }

    #[test]
    fn test_all_sites_findable() {
        for (id, _, _, _, _) in SITES {
            assert!(find_site(id).is_some(), "Site {} should be findable", id);
        }
    }

    #[test]
    fn test_koun_excluded() {
        assert!(find_site("KOUN").is_none(), "KOUN (research) should not be in operational table");
    }

    #[test]
    fn test_nearest_to_okc_is_ktlx() {
        // lat 35.2, lon -97.4 should resolve to KTLX, not KOUN (which is excluded)
        let nearest = SITES.iter()
            .min_by(|a, b| {
                let da = (a.2 - 35.2_f64).powi(2) + (a.3 - (-97.4_f64)).powi(2);
                let db = (b.2 - 35.2_f64).powi(2) + (b.3 - (-97.4_f64)).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        assert_eq!(nearest.0, "KTLX");
    }
}
